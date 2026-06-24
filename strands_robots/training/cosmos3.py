"""Cosmos3 trainer - drives cosmos_framework's SFT pipeline AS A LIBRARY.

Cosmos3 has the most distinct pipeline of the three backends, and is the reason
the :class:`Trainer` ABC has optional ``prepare``/``export`` hooks. NVIDIA's
``cosmos_framework`` is an internal package whose public entry points are
argv-parsing ``main()`` functions (hydra / tyro), so we drive it the
library way: **import the module and call its ``main()``** with a controlled
``argv`` LIST (via :func:`~strands_robots.training._inproc.call_module_main`).
No ``subprocess`` is spawned and no ``torchrun`` binary is used.

* **prepare()** - import ``cosmos_framework.scripts.convert_model_to_dcp`` and
  call its ``main()`` to convert the base HF checkpoint to PyTorch DCP.
* **train()** - import ``cosmos_framework.scripts.train`` and call its
  ``main()``; for multi-GPU/-node we wrap that call in torch's programmatic
  ``elastic_launch`` (the API behind ``torchrun``) so workers are spawned by the
  torch elastic agent, each calling ``main()`` in-process. The TOML recipe
  selects a registered experiment + scalar knobs; ``TrainSpec`` fields become
  Hydra ``key.path=value`` tail overrides (which win over the TOML).
* **export()** - import ``cosmos_framework.scripts.export_model`` and call its
  ``main()`` to convert the trained DCP back to HF safetensors.

Why import-and-call (not ``subprocess``/``torchrun``)
-----------------------------------------------------
The old implementation assembled string ``argv`` (partly from caller-controlled
``TrainSpec.extra`` -> Hydra ``key=value`` tail) and handed it to
``subprocess.run`` / ``torchrun``. We now build argv **lists** (never shell
strings), gate every passthrough Hydra key through
:func:`~strands_robots.training._inproc.safe_flag_key`, and invoke the cosmos
scripts' ``main()`` in this interpreter via ``call_module_main`` /
``elastic_launch`` - external input is never interpreted by a shell.

Multi-node HSDP maps ``num_nodes`` ->
``model.config.parallelism.data_parallel_replicate_degree`` (intra-node shard
stays at ``nproc_per_node``). 8xH100 80GB is the tested floor.

The cosmos_framework checkout is resolved from ``COSMOS_ROOT`` env var or
``extra['cosmos_root']``; the SFT recipe TOML from ``extra['sft_toml']``.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any

from strands_robots.training._inproc import (
    call_module_main,
    elastic_launch_callable,
    filter_safe_extra,
)
from strands_robots.training.base import Trainer, TrainResult, TrainSpec

logger = logging.getLogger(__name__)

_SUPPORTED_METHODS = {"full", "lora"}

_CONVERT_MODULE = "cosmos_framework.scripts.convert_model_to_dcp"
_TRAIN_MODULE = "cosmos_framework.scripts.train"
_EXPORT_MODULE = "cosmos_framework.scripts.export_model"


class Cosmos3Trainer(Trainer):
    """Post-tune an NVIDIA Cosmos3 policy via the cosmos_framework library.

    Args:
        cosmos_root: Path to the cosmos-framework checkout. Falls back to the
            ``COSMOS_ROOT`` env var, then ``TrainSpec.extra['cosmos_root']``.
            Added to ``sys.path`` so ``import cosmos_framework`` resolves.
    """

    def __init__(self, cosmos_root: str | None = None, **kwargs: Any) -> None:
        self.cosmos_root = cosmos_root or os.environ.get("COSMOS_ROOT")

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

    def _ensure_importable(self, spec: TrainSpec) -> None:
        """Put the resolved checkout on sys.path so ``import cosmos_framework`` works."""
        root = self._resolve_cosmos_root(spec)
        if root and root not in sys.path and os.path.isdir(os.path.join(root, "cosmos_framework")):
            sys.path.insert(0, root)

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
        Imports ``convert_model_to_dcp`` and calls its ``main()`` (no subprocess).
        """
        root = self._resolve_cosmos_root(spec)
        if not root:
            return
        self._ensure_importable(spec)
        dcp = self._dcp_path(spec)
        if os.path.isdir(os.path.join(dcp, "model")):
            logger.info("Cosmos3 DCP base already present at %s; skipping convert", dcp)
            return

        argv = self.convert_argv(spec)
        os.makedirs(os.path.dirname(os.path.abspath(dcp)) or ".", exist_ok=True)
        logger.info("Cosmos3Trainer converting base->DCP in-process: argv=%s", argv)
        call_module_main(_CONVERT_MODULE, argv, cwd=root, env={"PYTHONUNBUFFERED": "1"})

    def convert_argv(self, spec: TrainSpec) -> list[str]:
        """Argument LIST for convert_model_to_dcp: ``<ckpt> -o <dcp>`` (pure)."""
        return [spec.base_model, "-o", self._dcp_path(spec)]

    def build_argv(self, spec: TrainSpec) -> list[str]:
        """Argument LIST for cosmos_framework.scripts.train (pure, no launcher).

        Returns ``[--sft-toml=..., --, <Hydra tail overrides>]`` - the cosmos
        train argv passed to its ``main()``. Building a list (not a shell string)
        plus the safe-key gate on the Hydra tail removes the old injection surface.
        """
        argv = [f"--sft-toml={spec.extra['sft_toml']}"]

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
        _consumed = {"cosmos_root", "sft_toml", "dcp_path", "export_dir", "rdzv_endpoint"}
        safe, rejected = filter_safe_extra(spec.extra, _consumed)
        for key, value in safe.items():
            tail.append(f"{key}={value}")
        for key in rejected:
            logger.warning(
                "Cosmos3Trainer: ignoring unsafe extra key %r (not [A-Za-z0-9_.-]+).",
                key,
            )

        argv.extend(tail)
        return argv

    def export(self, spec: TrainSpec, checkpoint_dir: str) -> str:
        """Convert the trained DCP checkpoint back to HF safetensors (in-process).

        Returns a directory that ``create_policy`` can load. If cosmos_root is
        unavailable, falls back to the default passthrough. Imports
        ``export_model`` and calls its ``main()`` (no subprocess).
        """
        root = self._resolve_cosmos_root(spec)
        out = spec.extra.get(
            "export_dir", os.path.join(spec.output_dir, "_exported")
        )
        if not root:
            return checkpoint_dir
        self._ensure_importable(spec)
        argv = self.export_argv(spec, checkpoint_dir, out)
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        logger.info("Cosmos3Trainer exporting DCP->safetensors in-process: argv=%s", argv)
        try:
            call_module_main(_EXPORT_MODULE, argv, cwd=root, env={"PYTHONUNBUFFERED": "1"})
        except BaseException as e:  # noqa: BLE001 - export is best-effort; fall back
            logger.error("Cosmos3 export failed (%s); returning DCP checkpoint dir", e)
            return checkpoint_dir
        return out

    def export_argv(self, spec: TrainSpec, checkpoint_dir: str, out: str) -> list[str]:
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

        self._ensure_importable(spec)
        argv = self.build_argv(spec)
        job_id = f"cosmos3-{int(time.time())}"
        log_path = os.path.join(parent, f"{os.path.basename(spec.output_dir)}.{job_id}.log")
        root = self._resolve_cosmos_root(spec)
        nproc = self._nproc(spec)

        logger.info(
            "Cosmos3Trainer launching train() in-process: nproc=%d nnodes=%d steps=%d",
            nproc, max(1, spec.num_nodes), spec.steps,
        )

        train_error: BaseException | None = None
        try:
            if nproc > 1 or spec.num_nodes > 1:
                # Multi-GPU/-node: torch elastic agent spawns workers; each
                # imports cosmos train and calls its main() with the same argv.
                rdzv = spec.extra.get("rdzv_endpoint", "") if spec.num_nodes > 1 else ""
                elastic_launch_callable(
                    _cosmos_worker,
                    nproc_per_node=nproc,
                    nnodes=max(1, spec.num_nodes),
                    rdzv_endpoint=rdzv,
                    run_id=job_id,
                    fn_args=(argv, root, log_path),
                )
            else:
                call_module_main(
                    _TRAIN_MODULE, argv, cwd=root,
                    env={"PYTHONUNBUFFERED": "1"}, log_path=log_path,
                )
        except BaseException as e:  # noqa: BLE001 - convert ANY failure to a result
            train_error = e
            logger.error("Cosmos3Trainer in-process train failed: %s", e)

        ckpt = spec.output_dir
        if train_error is not None:
            return TrainResult(
                status="error", job_id=job_id, checkpoint_dir=ckpt,
                message=f"{_TRAIN_MODULE}.main() raised "
                        f"{type(train_error).__name__}: {train_error}; see {log_path}",
            )
        return TrainResult(
            status="success", job_id=job_id, checkpoint_dir=ckpt,
            message=f"Cosmos3 SFT complete (in-process); log: {log_path}",
        )


def _cosmos_worker(argv: list[str], cwd: str, log_path: str) -> None:
    """elastic_launch worker: import cosmos train and call its main() in this worker.

    Runs in a torch-spawned worker process (one per GPU). torch sets RANK /
    LOCAL_RANK / WORLD_SIZE; cosmos's HSDP/torch.distributed init reads them.
    The checkout is re-added to sys.path here (a fresh spawned interpreter).
    Only local rank 0 tees to the shared log file to avoid interleaved writes.
    """
    if cwd and cwd not in sys.path and os.path.isdir(os.path.join(cwd, "cosmos_framework")):
        sys.path.insert(0, cwd)
    is_rank0 = os.environ.get("LOCAL_RANK", "0") == "0"
    call_module_main(
        _TRAIN_MODULE, argv, cwd=cwd, env={"PYTHONUNBUFFERED": "1"},
        log_path=log_path if is_rank0 else None,
    )
