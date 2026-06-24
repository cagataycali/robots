"""Cosmos3 trainer — wrapper over cosmos_framework's SFT pipeline.

Cosmos3 has the most distinct pipeline of the three backends, and is the reason
the :class:`Trainer` ABC has optional ``prepare``/``export`` hooks:

* **prepare()** — the base HF checkpoint MUST be converted to PyTorch DCP
  (``cosmos_framework.scripts.convert_model_to_dcp``) before training.
  LeRobot/GR00T need no such step.
* **train()** — drives ``cosmos_framework.scripts.train`` via ``torchrun``.
  A TOML recipe selects a registered experiment + scalar knobs; ``TrainSpec``
  fields become Hydra ``key.path=value`` tail overrides (which win over the
  TOML). The ``torchrun`` distributed launcher is invoked from Python via
  :mod:`torch.distributed.run` (no shell, no extra interpreter).
* **export()** — the trained DCP is converted back to HF safetensors via
  ``cosmos_framework.scripts.export_model`` so ``create_policy`` can consume
  it.

Multi-node HSDP maps ``num_nodes`` →
``model.config.parallelism.data_parallel_replicate_degree`` (intra-node shard
stays at ``nproc_per_node``). 8×H100 80 GB is the tested floor.

The cosmos_framework checkout is resolved from ``COSMOS_ROOT`` env var or
``extra['cosmos_root']``; the SFT recipe TOML from ``extra['sft_toml']``.

Install cosmos-framework from source (per its README) and ensure
``cosmos_framework`` is importable in the active Python — this trainer drives
it as a Python library, NOT as a subprocess invoking another interpreter.
"""

from __future__ import annotations

import importlib
import logging
import os
import time
from typing import Any

from strands_robots.training.base import Trainer, TrainResult, TrainSpec

logger = logging.getLogger(__name__)

_SUPPORTED_METHODS = {"full", "lora"}

_INSTALL_HINT = (
    "cosmos-framework is not importable from this interpreter. "
    "Install it from source (see https://github.com/nvidia-cosmos/cosmos-framework "
    "or the path passed via cosmos_root / COSMOS_ROOT) into the *same* Python "
    "that imports strands_robots — e.g. `pip install -e $COSMOS_ROOT`."
)


def _import_cosmos_module(qualname: str):
    """Import ``cosmos_framework.<qualname>`` or raise a helpful ImportError.

    ``qualname`` is e.g. ``scripts.convert_model_to_dcp`` /
    ``scripts.train`` / ``scripts.export_model``. We resolve as a Python
    library so the trainer runs in-process — no nested ``python`` invocation.
    """
    full = f"cosmos_framework.{qualname}"
    try:
        return importlib.import_module(full)
    except ImportError as e:  # pragma: no cover - exercised in integration
        raise ImportError(f"{_INSTALL_HINT} (failed to import {full})") from e


class Cosmos3Trainer(Trainer):
    """Post-tune an NVIDIA Cosmos3 policy via cosmos_framework SFT.

    Args:
        cosmos_root: Path to the cosmos-framework checkout (used for
            ``torchrun``'s ``cwd`` so relative recipe/config paths resolve).
            Falls back to the ``COSMOS_ROOT`` env var, then
            ``TrainSpec.extra['cosmos_root']``. The package itself is loaded
            as a Python library via :func:`importlib.import_module` from the
            active interpreter — install from source per cosmos-framework's
            README; ``COSMOS_ROOT`` is for runtime config resolution, not the
            interpreter path.
    """

    def __init__(
        self,
        cosmos_root: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.cosmos_root = cosmos_root or os.environ.get("COSMOS_ROOT")

    @property
    def provider_name(self) -> str:
        return "cosmos3"

    @property
    def hardware_floor(self) -> dict[str, Any]:
        # SFT tested on 8xH100 80GB; HSDP multi-node beyond.
        return {"min_gpus": 8, "min_vram_gb": 80, "multinode": True}

    def _resolve_cosmos_root(self, spec: TrainSpec) -> str | None:
        return self.cosmos_root or spec.extra.get("cosmos_root")

    def _dcp_path(self, spec: TrainSpec) -> str:
        """Where prepare() writes (and train() reads) the DCP base checkpoint."""
        return str(spec.extra.get("dcp_path", os.path.join(spec.output_dir, "_dcp_base")))

    def _nproc(self, spec: TrainSpec) -> int:
        return max(1, spec.num_gpus)

    def validate(self, spec: TrainSpec) -> list[str]:
        problems: list[str] = self._security_problems(spec)

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
                f"unsupported method '{spec.method}' for Cosmos3 (expected one of {sorted(_SUPPORTED_METHODS)})"
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
                "cosmos-framework checkout not found; set COSMOS_ROOT, pass cosmos_root=..., or extra['cosmos_root']"
            )
        elif not os.path.isdir(os.path.join(root, "cosmos_framework")):
            problems.append(f"cosmos_framework package not found under cosmos_root={root}")

        return problems

    def prepare(self, spec: TrainSpec) -> None:
        """Convert the base HF checkpoint to DCP (required before training).

        Skips if the DCP target already exists (idempotent). Drives
        ``cosmos_framework.scripts.convert_model_to_dcp`` as a Python library
        — same interpreter, no subprocess.
        """
        root = self._resolve_cosmos_root(spec)
        if not root:
            return
        dcp = self._dcp_path(spec)
        if os.path.isdir(os.path.join(dcp, "model")):
            logger.info("Cosmos3 DCP base already present at %s; skipping convert", dcp)
            return

        os.makedirs(os.path.dirname(os.path.abspath(dcp)) or ".", exist_ok=True)
        logger.info("Cosmos3Trainer converting base->DCP: %s -> %s", spec.base_model, dcp)
        mod = _import_cosmos_module("scripts.convert_model_to_dcp")
        main = getattr(mod, "main", None) or getattr(mod, "convert", None)
        if main is None:
            raise ImportError("cosmos_framework.scripts.convert_model_to_dcp has no main()/convert() entrypoint")
        prev_cwd = os.getcwd()
        try:
            if os.path.isdir(root):
                os.chdir(root)
            main([spec.base_model, "-o", dcp])
        finally:
            os.chdir(prev_cwd)

    def convert_command(self, spec: TrainSpec) -> list[str]:
        """Argv parity helper: what ``convert_model_to_dcp`` would be called with.

        Used by parity tests + ``--dry-run`` previews. The trainer no longer
        spawns a subprocess for the conversion — it imports the module — but
        keeping a faithful argv lets the parity suite still assert flag drift.
        """
        return [
            "python",
            "-m",
            "cosmos_framework.scripts.convert_model_to_dcp",
            spec.base_model,
            "-o",
            self._dcp_path(spec),
        ]

    def build_command(self, spec: TrainSpec) -> list[str]:
        """Argv for ``cosmos_framework.scripts.train`` (driven via torch.distributed.run).

        Kept as an argv list for parity tests and dry-run preview; at run
        time we invoke :mod:`torch.distributed.run` from Python instead of
        spawning a ``torchrun`` subprocess.
        """
        nproc = self._nproc(spec)
        launcher = [
            "torchrun",
            f"--nproc_per_node={nproc}",
            f"--nnodes={max(1, spec.num_nodes)}",
            "-m",
            "cosmos_framework.scripts.train",
        ]
        cmd = [*launcher, f"--sft-toml={spec.extra['sft_toml']}"]

        # Everything after `--` is a Hydra tail override (wins over the TOML).
        tail: list[str] = ["--"]
        tail.append(f"trainer.max_iter={spec.steps}")
        tail.append(f"checkpoint.save_iter={spec.save_freq}")
        tail.append(f"optimizer.lr={spec.learning_rate}")
        tail.append(f"checkpoint.load_path={self._dcp_path(spec)}")
        tail.append(f"dataloader_train.max_samples_per_batch={spec.global_batch_size}")
        if spec.num_nodes > 1:
            tail.append(f"model.config.parallelism.data_parallel_replicate_degree={spec.num_nodes}")
        if spec.seed is not None:
            tail.append(f"trainer.seed={spec.seed}")

        _consumed = {"cosmos_root", "sft_toml", "dcp_path"}
        for key, value in spec.extra.items():
            if key in _consumed:
                continue
            tail.append(f"{key}={value}")

        cmd.extend(tail)
        return cmd

    def export(self, spec: TrainSpec, checkpoint_dir: str) -> str:
        """Convert the trained DCP checkpoint back to HF safetensors.

        Drives ``cosmos_framework.scripts.export_model`` as a Python library.
        Falls back to the input ``checkpoint_dir`` (default passthrough) if
        ``cosmos_root`` is unavailable.
        """
        root = self._resolve_cosmos_root(spec)
        out = spec.extra.get("export_dir", os.path.join(spec.output_dir, "_exported"))
        if not root:
            return checkpoint_dir
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        logger.info("Cosmos3Trainer exporting DCP->safetensors: %s -> %s", checkpoint_dir, out)
        try:
            mod = _import_cosmos_module("scripts.export_model")
            main = getattr(mod, "main", None) or getattr(mod, "export", None)
            if main is None:
                logger.warning(
                    "cosmos_framework.scripts.export_model has no main()/export() entrypoint; "
                    "falling back to checkpoint_dir"
                )
                return checkpoint_dir
            prev_cwd = os.getcwd()
            try:
                if os.path.isdir(root):
                    os.chdir(root)
                main([f"--checkpoint-path={checkpoint_dir}", f"--output-dir={out}"])
            finally:
                os.chdir(prev_cwd)
        except Exception as e:  # noqa: BLE001 — best-effort export; surface in log
            logger.warning("Cosmos3 export failed: %s; falling back to %s", e, checkpoint_dir)
            return checkpoint_dir
        return out

    def export_command(self, spec: TrainSpec, checkpoint_dir: str, out: str) -> list[str]:
        """Argv parity helper for ``export_model`` (see :meth:`build_command`)."""
        return [
            "python",
            "-m",
            "cosmos_framework.scripts.export_model",
            f"--checkpoint-path={checkpoint_dir}",
            f"--output-dir={out}",
        ]

    def train(self, spec: TrainSpec) -> TrainResult:
        problems = self.validate(spec)
        if problems:
            return TrainResult(
                status="error",
                job_id="",
                message="validation failed: " + "; ".join(problems),
            )

        parent = os.path.dirname(os.path.abspath(spec.output_dir)) or "."
        os.makedirs(parent, exist_ok=True)

        # prepare(): convert base -> DCP (idempotent), as a Python library call.
        try:
            self.prepare(spec)
        except ImportError as e:
            return TrainResult(
                status="error",
                job_id="",
                message=f"DCP conversion (prepare) failed: {e}",
            )
        except Exception as e:  # noqa: BLE001 — prepare is user-library code
            return TrainResult(
                status="error",
                job_id="",
                message=f"DCP conversion (prepare) failed: {e}",
            )

        job_id = f"cosmos3-{int(time.time())}"
        log_path = os.path.join(parent, f"{os.path.basename(spec.output_dir)}.{job_id}.log")
        root = self._resolve_cosmos_root(spec)

        # Verify the train entrypoint is importable BEFORE we spin up torchrun.
        try:
            _import_cosmos_module("scripts.train")
        except ImportError as e:
            return TrainResult(
                status="error",
                job_id=job_id,
                message=str(e),
            )

        # Build the argv that torch.distributed.run will execute as if from a shell.
        train_argv = self._distributed_argv(spec)

        logger.info("Cosmos3Trainer launching (torch.distributed.run): %s", " ".join(train_argv))
        prev_env = {k: os.environ.get(k) for k in ("PYTHONUNBUFFERED",)}
        os.environ["PYTHONUNBUFFERED"] = "1"
        prev_cwd = os.getcwd()
        rc = 0
        try:
            if root and os.path.isdir(root):
                os.chdir(root)
            # Redirect stdout/stderr to log_path while torch.distributed.run executes.
            with open(log_path, "w", encoding="utf-8") as logf:
                import contextlib

                from torch.distributed import run as torch_run

                with contextlib.redirect_stdout(logf), contextlib.redirect_stderr(logf):
                    try:
                        torch_run.main(train_argv)
                    except SystemExit as se:
                        rc = int(se.code) if se.code is not None else 0
        except ImportError as e:
            return TrainResult(
                status="error",
                job_id=job_id,
                message=f"torch.distributed.run not available ({e}); install torch with distributed support",
            )
        except Exception as e:  # noqa: BLE001 — surface launcher errors
            return TrainResult(
                status="error",
                job_id=job_id,
                message=f"cosmos_framework.scripts.train raised: {e}; see {log_path}",
            )
        finally:
            os.chdir(prev_cwd)
            for k, v in prev_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        ckpt = spec.output_dir
        if rc != 0:
            return TrainResult(
                status="error",
                job_id=job_id,
                checkpoint_dir=ckpt,
                message=f"cosmos_framework.scripts.train exited {rc}; see {log_path}",
            )
        return TrainResult(
            status="success",
            job_id=job_id,
            checkpoint_dir=ckpt,
            message=f"Cosmos3 SFT complete; log: {log_path}",
        )

    def _distributed_argv(self, spec: TrainSpec) -> list[str]:
        """Build the argv for :func:`torch.distributed.run.main`.

        Equivalent to the ``torchrun --nproc_per_node=N --nnodes=M -m
        cosmos_framework.scripts.train ...`` argv, but consumed by
        :mod:`torch.distributed.run` in-process (no subprocess for ``torchrun``
        itself; it still spawns worker processes as torch.distributed requires).
        """
        nproc = self._nproc(spec)
        argv: list[str] = [
            f"--nproc_per_node={nproc}",
            f"--nnodes={max(1, spec.num_nodes)}",
            "-m",
            "cosmos_framework.scripts.train",
            f"--sft-toml={spec.extra['sft_toml']}",
            "--",
            f"trainer.max_iter={spec.steps}",
            f"checkpoint.save_iter={spec.save_freq}",
            f"optimizer.lr={spec.learning_rate}",
            f"checkpoint.load_path={self._dcp_path(spec)}",
            f"dataloader_train.max_samples_per_batch={spec.global_batch_size}",
        ]
        if spec.num_nodes > 1:
            argv.append(f"model.config.parallelism.data_parallel_replicate_degree={spec.num_nodes}")
        if spec.seed is not None:
            argv.append(f"trainer.seed={spec.seed}")
        _consumed = {"cosmos_root", "sft_toml", "dcp_path"}
        for key, value in spec.extra.items():
            if key in _consumed:
                continue
            argv.append(f"{key}={value}")
        return argv
