"""Cosmos3 trainer - wrapper over cosmos_framework's SFT pipeline.

Cosmos3 has the most distinct pipeline of the three backends, and is the reason
the :class:`Trainer` ABC has optional ``prepare``/``export`` hooks:

* **prepare()** - the base HF checkpoint MUST be converted to PyTorch DCP
  (``python -m cosmos_framework.scripts.convert_model_to_dcp <ckpt> -o <dcp>``)
  before training. LeRobot/GR00T need no such step.
* **train()** - ``torchrun --nproc_per_node=N -m cosmos_framework.scripts.train
  --sft-toml=<recipe.toml> -- <Hydra tail overrides>``. The TOML recipe selects
  a registered experiment + scalar knobs; ``TrainSpec`` fields become Hydra
  ``key.path=value`` tail overrides (which win over the TOML).
* **export()** - the trained DCP is converted back to HF safetensors
  (``python -m cosmos_framework.scripts.export_model``) so ``create_policy`` can
  consume it.

Multi-node HSDP maps ``num_nodes`` ->
``model.config.parallelism.data_parallel_replicate_degree`` (intra-node shard
stays at ``nproc_per_node``). 8xH100 80GB is the tested floor.

The cosmos_framework checkout is resolved from ``COSMOS_ROOT`` env var or
``extra['cosmos_root']``; the SFT recipe TOML from ``extra['sft_toml']``.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from typing import Any

from strands_robots.training.base import Trainer, TrainResult, TrainSpec

logger = logging.getLogger(__name__)

_SUPPORTED_METHODS = {"full", "lora"}


class Cosmos3Trainer(Trainer):
    """Post-tune an NVIDIA Cosmos3 policy via cosmos_framework SFT.

    Args:
        cosmos_root: Path to the cosmos-framework checkout. Falls back to the
            ``COSMOS_ROOT`` env var, then ``TrainSpec.extra['cosmos_root']``.
        python_executable: Interpreter for subprocesses (default current).
    """

    def __init__(
        self,
        cosmos_root: str | None = None,
        python_executable: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.cosmos_root = cosmos_root or os.environ.get("COSMOS_ROOT")
        self.python_executable = python_executable or sys.executable

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
        return str(spec.extra.get("dcp_path", os.path.join(spec.output_dir, "_dcp_base")))

    def _nproc(self, spec: TrainSpec) -> int:
        return max(1, spec.num_gpus)

    # ---- ABC ---------------------------------------------------------------

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

        Skips if the DCP target already exists (idempotent). Honored only when
        a cosmos_root is resolvable; otherwise train()'s validate() reports it.
        """
        root = self._resolve_cosmos_root(spec)
        if not root:
            return
        dcp = self._dcp_path(spec)
        if os.path.isdir(os.path.join(dcp, "model")):
            logger.info("Cosmos3 DCP base already present at %s; skipping convert", dcp)
            return

        cmd = self.convert_command(spec)
        os.makedirs(os.path.dirname(os.path.abspath(dcp)) or ".", exist_ok=True)
        logger.info("Cosmos3Trainer converting base->DCP: %s", " ".join(cmd))
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        subprocess.run(cmd, cwd=root, env=env, check=True)  # noqa: S603 - argv values validated by validate()/_security_problems before train() builds the command; list form, no shell

    def convert_command(self, spec: TrainSpec) -> list[str]:
        """python -m cosmos_framework.scripts.convert_model_to_dcp <ckpt> -o <dcp>."""
        return [
            self.python_executable,
            "-m",
            "cosmos_framework.scripts.convert_model_to_dcp",
            spec.base_model,
            "-o",
            self._dcp_path(spec),
        ]

    def build_command(self, spec: TrainSpec) -> list[str]:
        """torchrun -m cosmos_framework.scripts.train --sft-toml=... -- <overrides>."""
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
        # Per-step batch lives in the experiment; expose the count knob.
        tail.append(f"dataloader_train.max_samples_per_batch={spec.global_batch_size}")
        # Multi-node HSDP: replicate degree = number of nodes.
        if spec.num_nodes > 1:
            tail.append(f"model.config.parallelism.data_parallel_replicate_degree={spec.num_nodes}")
        if spec.seed is not None:
            tail.append(f"trainer.seed={spec.seed}")

        # Extra Hydra overrides passthrough (skip consumed keys).
        _consumed = {"cosmos_root", "sft_toml", "dcp_path"}
        for key, value in spec.extra.items():
            if key in _consumed:
                continue
            # Dotted keys are treated as Hydra overrides verbatim.
            tail.append(f"{key}={value}")

        cmd.extend(tail)
        return cmd

    def export(self, spec: TrainSpec, checkpoint_dir: str) -> str:
        """Convert the trained DCP checkpoint back to HF safetensors.

        Returns a directory that ``create_policy`` can load. If cosmos_root is
        unavailable, falls back to the default passthrough.
        """
        root = self._resolve_cosmos_root(spec)
        out = spec.extra.get("export_dir", os.path.join(spec.output_dir, "_exported"))
        if not root:
            return checkpoint_dir
        cmd = self.export_command(spec, checkpoint_dir, out)
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        logger.info("Cosmos3Trainer exporting DCP->safetensors: %s", " ".join(cmd))
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        proc = subprocess.run(cmd, cwd=root, env=env, check=False)  # noqa: S603 - argv values validated by validate()/_security_problems before train() builds the command; list form, no shell
        return out if proc.returncode == 0 else checkpoint_dir

    def export_command(self, spec: TrainSpec, checkpoint_dir: str, out: str) -> list[str]:
        """python -m cosmos_framework.scripts.export_model (DCP -> safetensors)."""
        return [
            self.python_executable,
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

        # prepare(): convert base -> DCP (idempotent).
        try:
            self.prepare(spec)
        except subprocess.CalledProcessError as e:
            return TrainResult(
                status="error",
                job_id="",
                message=f"DCP conversion (prepare) failed: {e}",
            )

        cmd = self.build_command(spec)
        job_id = f"cosmos3-{int(time.time())}"
        log_path = os.path.join(parent, f"{os.path.basename(spec.output_dir)}.{job_id}.log")
        root = self._resolve_cosmos_root(spec)

        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")

        logger.info("Cosmos3Trainer launching: %s", " ".join(cmd))
        try:
            with open(log_path, "w", encoding="utf-8") as logf:
                proc = subprocess.run(  # noqa: S603 - argv values validated by validate()/_security_problems before train() builds the command; list form, no shell
                    cmd,
                    cwd=root,
                    env=env,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
        except FileNotFoundError as e:
            return TrainResult(
                status="error",
                job_id=job_id,
                message=f"torchrun not found ({e}); is cosmos-framework's train env active?",
            )

        ckpt = spec.output_dir
        if proc.returncode != 0:
            return TrainResult(
                status="error",
                job_id=job_id,
                checkpoint_dir=ckpt,
                message=f"cosmos_framework.scripts.train exited {proc.returncode}; see {log_path}",
            )
        return TrainResult(
            status="success",
            job_id=job_id,
            checkpoint_dir=ckpt,
            message=f"Cosmos3 SFT complete; log: {log_path}",
        )
