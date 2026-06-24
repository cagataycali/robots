"""Cosmos3 trainer - in-process wrapper over cosmos_framework's SFT pipeline.

Cosmos3 has the most distinct pipeline of the three backends, and is the reason
the :class:`Trainer` ABC has optional ``prepare``/``export`` hooks. All three
of its stages now run **in-process** (no ``subprocess``, no ``torchrun`` binary):

* **prepare()** - the base HF checkpoint MUST be converted to PyTorch DCP. We
  run ``cosmos_framework.scripts.convert_model_to_dcp`` IN-PROCESS via
  :func:`runpy.run_module` with a controlled argv list.
* **train()** - ``cosmos_framework.scripts.train --sft-toml=<recipe.toml> --
  <Hydra tail overrides>``, launched via torch's programmatic ``elastic_launch``
  (the API behind ``torchrun``) so workers are spawned by the torch elastic
  agent, NOT a ``torchrun`` command line. The TOML recipe selects a registered
  experiment + scalar knobs; ``TrainSpec`` fields become Hydra
  ``key.path=value`` tail overrides (which win over the TOML).
* **export()** - the trained DCP is converted back to HF safetensors via
  ``cosmos_framework.scripts.export_model`` (in-process) so ``create_policy``
  can consume it.

Why not ``subprocess`` + ``torchrun``
-------------------------------------
The old implementation assembled string ``argv`` (partly from caller-controlled
``TrainSpec.extra`` -> Hydra ``key=value`` tail) and handed it to
``subprocess.run`` / ``torchrun``. Building a command line from external input
for a spawned interpreter is a needless injection / arbitrary-flag surface. We
now build argv **lists** (never shell strings), gate every passthrough Hydra
key through :func:`~strands_robots.training._inproc.safe_flag_key`, and execute
the cosmos scripts via :mod:`runpy` / ``elastic_launch`` - external input is
never interpreted by a shell.

Multi-node HSDP maps ``num_nodes`` ->
``model.config.parallelism.data_parallel_replicate_degree`` (intra-node shard
stays at ``nproc_per_node``). 8xH100 80GB is the tested floor.

The cosmos_framework checkout is resolved from ``COSMOS_ROOT`` env var or
``extra['cosmos_root']``; the SFT recipe TOML from ``extra['sft_toml']``.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from strands_robots.training._inproc import (
    elastic_launch_callable,
    filter_safe_extra,
    run_python_module,
)
from strands_robots.training.base import Trainer, TrainResult, TrainSpec

logger = logging.getLogger(__name__)

_SUPPORTED_METHODS = {"full", "lora"}


class Cosmos3Trainer(Trainer):
    """Post-tune an NVIDIA Cosmos3 policy via cosmos_framework SFT (in-process).

    Args:
        cosmos_root: Path to the cosmos-framework checkout. Falls back to the
            ``COSMOS_ROOT`` env var, then ``TrainSpec.extra['cosmos_root']``.
        python_executable: Deprecated / ignored. Kept so existing callers
            constructing ``Cosmos3Trainer(python_executable=...)`` don't break;
            the cosmos scripts now run in THIS interpreter (or torch-spawned
            workers), so there is no child process to point at a different Python.
    """

    def __init__(
        self,
        cosmos_root: str | None = None,
        python_executable: str | None = None,  # noqa: ARG002 - back-compat shim, ignored
        **kwargs: Any,
    ) -> None:
        self.cosmos_root = cosmos_root or os.environ.get("COSMOS_ROOT")
        if python_executable is not None:
            logger.debug(
                "Cosmos3Trainer(python_executable=%r) is ignored: cosmos scripts "
                "now run in-process (no subprocess).",
                python_executable,
            )

    @property
    def provider_name(self) -> str:
        return "cosmos3"

    @property
    def hardware_floor(self) -> dict[str, Any]:
        # SFT tested on 8xH100 80GB; HSDP multi-node beyond.
        return {"min_gpus": 8, "min_vram_gb": 80, "multinode": True}

    # ---- helpers -----------------------------------------------------------

    def _resolve_cosmos_root(self, spec: TrainSpec) -> str | None:
        return self.cosmos_root or spec.extra.get("cosmos_root")

    def _dcp_path(self, spec: TrainSpec) -> str:
        """Where prepare() writes (and train() reads) the DCP base checkpoint."""
        return spec.extra.get(
            "dcp_path", os.path.join(spec.output_dir, "_dcp_base")
        )

    def _nproc(self, spec: TrainSpec) -> int:
        return max(1, spec.num_gpus)

    # ---- ABC ---------------------------------------------------------------

    def validate(self, spec: TrainSpec) -> list[str]:
        problems: list[str] = []

        if not spec.dataset_root:
            problems.append("dataset_root is required")
        elif not os.path.isfile(os.path.join(spec.dataset_root, "meta", "info.json")):
            problems.append(
                f"dataset_root is not a LeRobotDataset v3 root "
                f"(missing {os.path.join(spec.dataset_root, 'meta', 'info.json')})"
            )

        if not spec.base_model:
            problems.append("base_model is required (HF checkpoint to convert to DCP)")
        if not spec.output_dir:
            problems.append("output_dir is required")

        if spec.method not in _SUPPORTED_METHODS:
            problems.append(
                f"unsupported method '{spec.method}' for Cosmos3 "
                f"(expected one of {sorted(_SUPPORTED_METHODS)})"
            )
        if spec.steps <= 0:
            problems.append(f"steps must be > 0, got {spec.steps}")

        if not spec.extra.get("sft_toml"):
            problems.append(
                "Cosmos3 needs a recipe TOML; pass extra['sft_toml']=<path> "
                "(selects the registered experiment + scalar knobs)"
            )
        elif not os.path.isfile(spec.extra["sft_toml"]):
            problems.append(f"sft_toml does not exist: {spec.extra['sft_toml']}")

        root = self._resolve_cosmos_root(spec)
        if not root:
            problems.append(
                "cosmos-framework checkout not found; set COSMOS_ROOT, pass "
                "cosmos_root=..., or extra['cosmos_root']"
            )
        elif not os.path.isdir(os.path.join(root, "cosmos_framework")):
            problems.append(f"cosmos_framework package not found under cosmos_root={root}")

        return problems

    def prepare(self, spec: TrainSpec) -> None:
        """Convert the base HF checkpoint to DCP (required before training).

        Skips if the DCP target already exists (idempotent). Honored only when
        a cosmos_root is resolvable; otherwise train()'s validate() reports it.
        Runs ``convert_model_to_dcp`` IN-PROCESS via runpy (no subprocess).
        """
        root = self._resolve_cosmos_root(spec)
        if not root:
            return
        dcp = self._dcp_path(spec)
        if os.path.isdir(os.path.join(dcp, "model")):
            logger.info("Cosmos3 DCP base already present at %s; skipping convert", dcp)
            return

        args = self.convert_args(spec)
        os.makedirs(os.path.dirname(os.path.abspath(dcp)) or ".", exist_ok=True)
        logger.info("Cosmos3Trainer converting base->DCP in-process: args=%s", args)
        run_python_module(
            "cosmos_framework.scripts.convert_model_to_dcp",
            args, cwd=root, env={"PYTHONUNBUFFERED": "1"},
        )

    def convert_args(self, spec: TrainSpec) -> list[str]:
        """Argument LIST for convert_model_to_dcp: ``<ckpt> -o <dcp>`` (pure)."""
        return [spec.base_model, "-o", self._dcp_path(spec)]

    def build_args(self, spec: TrainSpec) -> list[str]:
        """Argument LIST for cosmos_framework.scripts.train (pure, no launcher).

        Returns ``[--sft-toml=..., --, <Hydra tail overrides>]`` - the cosmos
        train argv WITHOUT any launcher binary. Caller runs this via
        ``elastic_launch`` (workers) or in-process. Building a list (not a shell
        string) plus the safe-key gate on the Hydra tail removes the old
        injection surface.
        """
        args = [f"--sft-toml={spec.extra['sft_toml']}"]

        # Everything after `--` is a Hydra tail override (wins over the TOML).
        tail: list[str] = ["--"]
        tail.append(f"trainer.max_iter={spec.steps}")
        tail.append(f"checkpoint.save_iter={spec.save_freq}")
        tail.append(f"optimizer.lr={spec.learning_rate}")
        tail.append(f"checkpoint.load_path={self._dcp_path(spec)}")
        # Per-step batch lives in the experiment; expose the count knob.
        tail.append(f"dataloader_train.max_samples_per_batch={spec.global_batch_size}")
        # Multi-node HSDP: replicate degree = number of nodes.
        if spec.num_nodes > 1:
            tail.append(
                f"model.config.parallelism.data_parallel_replicate_degree={spec.num_nodes}"
            )
        if spec.seed is not None:
            tail.append(f"trainer.seed={spec.seed}")

        # Extra Hydra overrides passthrough - dotted keys are valid Hydra paths,
        # but gate them so a stray entry can't smuggle extra argv tokens.
        _consumed = {"cosmos_root", "sft_toml", "dcp_path", "export_dir"}
        safe, rejected = filter_safe_extra(spec.extra, _consumed)
        for key, value in safe.items():
            tail.append(f"{key}={value}")
        for key in rejected:
            logger.warning(
                "Cosmos3Trainer: ignoring unsafe extra key %r (not [A-Za-z0-9_.-]+).",
                key,
            )

        args.extend(tail)
        return args

    def export(self, spec: TrainSpec, checkpoint_dir: str) -> str:
        """Convert the trained DCP checkpoint back to HF safetensors (in-process).

        Returns a directory that ``create_policy`` can load. If cosmos_root is
        unavailable, falls back to the default passthrough.
        """
        root = self._resolve_cosmos_root(spec)
        out = spec.extra.get(
            "export_dir", os.path.join(spec.output_dir, "_exported")
        )
        if not root:
            return checkpoint_dir
        args = self.export_args(spec, checkpoint_dir, out)
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        logger.info("Cosmos3Trainer exporting DCP->safetensors in-process: args=%s", args)
        try:
            run_python_module(
                "cosmos_framework.scripts.export_model",
                args, cwd=root, env={"PYTHONUNBUFFERED": "1"},
            )
        except BaseException as e:  # noqa: BLE001 - export is best-effort; fall back
            logger.error("Cosmos3 export failed (%s); returning DCP checkpoint dir", e)
            return checkpoint_dir
        return out

    def export_args(self, spec: TrainSpec, checkpoint_dir: str, out: str) -> list[str]:
        """Argument LIST for export_model (DCP -> safetensors) (pure)."""
        return [
            f"--checkpoint-path={checkpoint_dir}",
            f"--output-dir={out}",
        ]

    def train(self, spec: TrainSpec) -> TrainResult:
        problems = self.validate(spec)
        if problems:
            return TrainResult(
                status="error", job_id="",
                message="validation failed: " + "; ".join(problems),
            )

        parent = os.path.dirname(os.path.abspath(spec.output_dir)) or "."
        os.makedirs(parent, exist_ok=True)

        # prepare(): convert base -> DCP (idempotent, in-process).
        try:
            self.prepare(spec)
        except BaseException as e:  # noqa: BLE001 - surface convert failure as result
            return TrainResult(
                status="error", job_id="",
                message=f"DCP conversion (prepare) failed: {e}",
            )

        args = self.build_args(spec)
        job_id = f"cosmos3-{int(time.time())}"
        log_path = os.path.join(parent, f"{os.path.basename(spec.output_dir)}.{job_id}.log")
        root = self._resolve_cosmos_root(spec)
        nproc = self._nproc(spec)

        logger.info(
            "Cosmos3Trainer launching in-process: nproc=%d nnodes=%d steps=%d",
            nproc, max(1, spec.num_nodes), spec.steps,
        )

        train_error: BaseException | None = None
        try:
            if nproc > 1 or spec.num_nodes > 1:
                # Multi-GPU/-node: torch elastic agent spawns workers; each runs
                # cosmos train in-process with the same argv list. The cosmos
                # train reads RANK/LOCAL_RANK/WORLD_SIZE that torch sets.
                rdzv = spec.extra.get("rdzv_endpoint", "") if spec.num_nodes > 1 else ""
                elastic_launch_callable(
                    _cosmos_worker,
                    nproc_per_node=nproc,
                    nnodes=max(1, spec.num_nodes),
                    rdzv_endpoint=rdzv,
                    run_id=job_id,
                    fn_args=(args, root, log_path),
                )
            else:
                run_python_module(
                    "cosmos_framework.scripts.train",
                    args, cwd=root, env={"PYTHONUNBUFFERED": "1"}, log_path=log_path,
                )
        except BaseException as e:  # noqa: BLE001 - convert ANY failure to a result
            train_error = e
            logger.error("Cosmos3Trainer in-process launch failed: %s", e)

        ckpt = spec.output_dir
        if train_error is not None:
            return TrainResult(
                status="error", job_id=job_id, checkpoint_dir=ckpt,
                message=f"cosmos_framework.scripts.train raised "
                        f"{type(train_error).__name__}: {train_error}; see {log_path}",
            )
        return TrainResult(
            status="success", job_id=job_id, checkpoint_dir=ckpt,
            message=f"Cosmos3 SFT complete (in-process); log: {log_path}",
        )


def _cosmos_worker(args: list[str], cwd: str, log_path: str) -> None:
    """elastic_launch worker: run cosmos_framework.scripts.train in this worker.

    Runs in a torch-spawned worker process (one per GPU). torch sets RANK /
    LOCAL_RANK / WORLD_SIZE; cosmos's HSDP/torch.distributed init reads them.
    Only local rank 0 tees to the shared log file to avoid interleaved writes.
    """
    is_rank0 = os.environ.get("LOCAL_RANK", "0") == "0"
    run_python_module(
        "cosmos_framework.scripts.train",
        args, cwd=cwd, env={"PYTHONUNBUFFERED": "1"},
        log_path=log_path if is_rank0 else None,
    )
