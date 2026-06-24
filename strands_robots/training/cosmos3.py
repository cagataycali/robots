"""Cosmos3 trainer - drives cosmos_framework's SFT pipeline AS A LIBRARY.

Cosmos3 has the most distinct pipeline of the three backends, and is the reason
the :class:`Trainer` ABC has optional ``prepare``/``export`` hooks. All three
of its stages call NVIDIA ``cosmos_framework``'s **own library functions**
directly (verified against the public ``github.com/NVIDIA/cosmos-framework``
checkout) - no ``subprocess`` is spawned and no ``torchrun`` binary is used:

* **prepare()** - import
  ``cosmos_framework.scripts.convert_model_to_dcp.convert_model_to_dcp`` and call
  it with a ``convert_model_to_dcp.Args`` object (a pydantic model:
  ``checkpoint=CheckpointOverrides(checkpoint_path=<hf>)``, ``output_path=<dcp>``).
* **train()** - build the merged ``Config`` via
  ``cosmos_framework.configs.toml_config.sft_config.load_experiment_from_toml``
  (TOML recipe + a list of Hydra ``key.path=value`` overrides), then call
  ``cosmos_framework.scripts.train.launch(config, args)`` directly. For
  multi-GPU/-node we wrap that call in torch's programmatic ``elastic_launch``
  (the API behind ``torchrun``) so workers are spawned by the torch elastic
  agent, each calling ``launch`` in-process.
* **export()** - import
  ``cosmos_framework.scripts.export_model.export_model`` and call it with an
  ``export_model.Args`` object (``checkpoint=CheckpointOverrides(checkpoint_path=
  <dcp>)``, ``output_dir=<hf>``).

Why call the functions directly (not ``subprocess`` / ``torchrun`` / argv)
--------------------------------------------------------------------------
The old implementation assembled string ``argv`` (partly from caller-controlled
``TrainSpec.extra`` -> Hydra ``key=value`` tail) and handed it to
``subprocess.run`` / ``torchrun``. ``cosmos_framework`` exposes real library
functions (``convert_model_to_dcp(args)`` / ``launch(config, args)`` /
``export_model(args)``), so we build their typed argument objects in Python and
call them. The only remaining strings are the Hydra override LIST passed to
``load_experiment_from_toml(extra_overrides=...)`` - each ``key=value`` entry is
gated by :func:`~strands_robots.training._inproc.safe_flag_key` so a stray
``extra`` entry can never smuggle extra tokens.

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
    call_callable,
    elastic_launch_callable,
    filter_safe_extra,
)
from strands_robots.training.base import Trainer, TrainResult, TrainSpec

logger = logging.getLogger(__name__)

_SUPPORTED_METHODS = {"full", "lora"}


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

    def build_overrides(self, spec: TrainSpec) -> list[str]:
        """Build the Hydra ``key.path=value`` override LIST (pure, testable).

        These are passed to ``load_experiment_from_toml(extra_overrides=...)`` -
        a LIST of ``key=value`` strings, NOT argv flags / a shell line. They are
        applied after the TOML so they win. Caller-supplied ``extra.*`` keys are
        gated by the safe-key allowlist so a stray entry can't inject tokens.
        """
        overrides = [
            f"trainer.max_iter={spec.steps}",
            f"checkpoint.save_iter={spec.save_freq}",
            f"optimizer.lr={spec.learning_rate}",
            f"checkpoint.load_path={self._dcp_path(spec)}",
            # Per-step batch lives in the experiment; expose the count knob.
            f"dataloader_train.max_samples_per_batch={spec.global_batch_size}",
        ]
        # Multi-node HSDP: replicate degree = number of nodes.
        if spec.num_nodes > 1:
            overrides.append(
                f"model.config.parallelism.data_parallel_replicate_degree={spec.num_nodes}"
            )
        if spec.seed is not None:
            overrides.append(f"trainer.seed={spec.seed}")

        # Extra Hydra overrides passthrough - dotted keys are valid Hydra paths,
        # but gate them so a stray entry can't smuggle extra tokens.
        _consumed = {"cosmos_root", "sft_toml", "dcp_path", "export_dir", "rdzv_endpoint"}
        safe, rejected = filter_safe_extra(spec.extra, _consumed)
        for key, value in safe.items():
            overrides.append(f"{key}={value}")
        for key in rejected:
            logger.warning(
                "Cosmos3Trainer: ignoring unsafe extra key %r (not [A-Za-z0-9_.-]+).",
                key,
            )
        return overrides

    def prepare(self, spec: TrainSpec) -> None:
        """Convert the base HF checkpoint to DCP (required before training).

        Skips if the DCP target already exists (idempotent). Honored only when
        a cosmos_root is resolvable; otherwise train()'s validate() reports it.
        Calls ``convert_model_to_dcp(Args(...))`` directly - no subprocess.
        """
        root = self._resolve_cosmos_root(spec)
        if not root:
            return
        self._ensure_importable(spec)
        dcp = self._dcp_path(spec)
        if os.path.isdir(os.path.join(dcp, "model")):
            logger.info("Cosmos3 DCP base already present at %s; skipping convert", dcp)
            return

        from cosmos_framework.inference.common.args import CheckpointOverrides
        from cosmos_framework.scripts.convert_model_to_dcp import Args, convert_model_to_dcp

        os.makedirs(os.path.dirname(os.path.abspath(dcp)) or ".", exist_ok=True)
        args = Args(
            checkpoint=CheckpointOverrides(checkpoint_path=spec.base_model),
            output_path=dcp,
        )
        logger.info("Cosmos3Trainer converting base->DCP in-process (library call)")
        call_callable(convert_model_to_dcp, args)

    def export(self, spec: TrainSpec, checkpoint_dir: str) -> str:
        """Convert the trained DCP checkpoint back to HF safetensors (in-process).

        Returns a directory that ``create_policy`` can load. If cosmos_root is
        unavailable, falls back to the default passthrough. Calls
        ``export_model(Args(...))`` directly - no subprocess.
        """
        root = self._resolve_cosmos_root(spec)
        out = spec.extra.get(
            "export_dir", os.path.join(spec.output_dir, "_exported")
        )
        if not root:
            return checkpoint_dir
        self._ensure_importable(spec)
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        try:
            from cosmos_framework.inference.common.args import CheckpointOverrides
            from cosmos_framework.scripts.export_model import Args, export_model

            args = Args(
                checkpoint=CheckpointOverrides(checkpoint_path=checkpoint_dir),
                output_dir=out,
            )
            logger.info("Cosmos3Trainer exporting DCP->safetensors in-process (library call)")
            call_callable(export_model, args)
        except BaseException as e:  # noqa: BLE001 - export is best-effort; fall back
            logger.error("Cosmos3 export failed (%s); returning DCP checkpoint dir", e)
            return checkpoint_dir
        return out

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
        sft_toml = spec.extra["sft_toml"]
        overrides = self.build_overrides(spec)
        job_id = f"cosmos3-{int(time.time())}"
        log_path = os.path.join(parent, f"{os.path.basename(spec.output_dir)}.{job_id}.log")
        nproc = self._nproc(spec)

        logger.info(
            "Cosmos3Trainer launching train() in-process: nproc=%d nnodes=%d steps=%d",
            nproc, max(1, spec.num_nodes), spec.steps,
        )

        train_error: BaseException | None = None
        try:
            if nproc > 1 or spec.num_nodes > 1:
                # Multi-GPU/-node: torch elastic agent spawns workers; each builds
                # the cosmos Config and calls launch() - Python objects, no argv.
                rdzv = spec.extra.get("rdzv_endpoint", "") if spec.num_nodes > 1 else ""
                elastic_launch_callable(
                    _cosmos_worker,
                    nproc_per_node=nproc,
                    nnodes=max(1, spec.num_nodes),
                    rdzv_endpoint=rdzv,
                    run_id=job_id,
                    fn_args=(sft_toml, overrides, log_path),
                )
            else:
                _run_cosmos_launch(sft_toml, overrides, log_path=log_path)
        except BaseException as e:  # noqa: BLE001 - convert ANY failure to a result
            train_error = e
            logger.error("Cosmos3Trainer in-process train failed: %s", e)

        ckpt = spec.output_dir
        if train_error is not None:
            return TrainResult(
                status="error", job_id=job_id, checkpoint_dir=ckpt,
                message=f"cosmos_framework train launch() raised "
                        f"{type(train_error).__name__}: {train_error}; see {log_path}",
            )
        return TrainResult(
            status="success", job_id=job_id, checkpoint_dir=ckpt,
            message=f"Cosmos3 SFT complete (in-process); log: {log_path}",
        )


def _run_cosmos_launch(sft_toml: str, overrides: list[str], *, log_path: str | None = None) -> None:
    """Build the cosmos Config from the TOML + overrides and call launch(config, args).

    Mirrors ``cosmos_framework/scripts/train.py``'s ``__main__`` (which has no
    reusable ``main()``): it builds ``config`` via ``load_experiment_from_toml``
    and calls ``launch(config, args)``. We do the same here, constructing the
    argparse-shaped ``args`` namespace with the non-deterministic defaults the
    script uses, so calling ``launch`` is behaviourally identical to the CLI -
    minus the process spawn and argv parse.
    """
    import argparse

    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
    from cosmos_framework.scripts.train import launch

    config = load_experiment_from_toml(sft_toml, extra_overrides=overrides)
    # The argparse.Namespace launch() reads: deterministic / attach_vscode_debugger
    # / dryrun / config (telemetry alias) / opts. Defaults match the CLI's
    # non-deterministic, non-debug, real-run path.
    args = argparse.Namespace(
        sft_toml=sft_toml,
        opts=list(overrides),
        deterministic=False,
        attach_vscode_debugger=False,
        dryrun=False,
        config=sft_toml,
    )
    call_callable(launch, config, args, log_path=log_path)


def _cosmos_worker(sft_toml: str, overrides: list[str], log_path: str) -> None:
    """elastic_launch worker: build the cosmos Config and call launch() in this worker.

    Runs in a torch-spawned worker process (one per GPU). torch sets RANK /
    LOCAL_RANK / WORLD_SIZE; cosmos's HSDP/distributed init reads them.
    Only local rank 0 tees to the shared log file to avoid interleaved writes.
    """
    is_rank0 = os.environ.get("LOCAL_RANK", "0") == "0"
    _run_cosmos_launch(sft_toml, overrides, log_path=log_path if is_rank0 else None)
