"""GR00T trainer - wrapper over Isaac-GR00T's ``launch_finetune.py``.

GR00T N1.7 ships its own post-training pipeline (NOT lerobot): a
``FinetuneConfig`` dataclass driven via tyro through
``gr00t/experiment/launch_finetune.py``, launched with plain ``python`` on a
single GPU (``CUDA_VISIBLE_DEVICES=0`` to avoid HF Trainer's DataParallel
wrap) or ``torchrun --nproc_per_node`` for multi-GPU - mirroring
``examples/finetune.sh``.

This adapter translates a provider-agnostic
:class:`~strands_robots.training.base.TrainSpec` into the ``launch_finetune.py``
argv. Mapping highlights:

* ``base_model``        -> ``--base_model_path``
* ``dataset_root``      -> ``--dataset_path``
* ``embodiment``        -> ``--embodiment_tag`` (REQUIRED by GR00T)
* ``steps``             -> ``--max_steps``
* ``global_batch_size`` -> ``--global_batch_size``
* ``learning_rate``     -> ``--learning_rate``
* ``save_freq``         -> ``--save_steps``
* ``resume``            -> ``--resume_from_checkpoint``
* ``tune`` dict         -> ``--tune_llm/--tune_visual/--tune_projector/--tune_diffusion_model``
* ``augmentation``      -> ``--random_rotation_angle`` / ``--color_jitter_params``
                           / ``--extra_augmentation_config`` (JSON)
* ``extra['modality_config_path']`` -> ``--modality_config_path``

GR00T checkpoints are HF-native, so :meth:`export` is the default passthrough.
The script path is resolved from the ``GR00T_ROOT`` env var (the Isaac-GR00T
checkout) or ``extra['groot_root']``.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from typing import Any

from strands_robots.training.base import Trainer, TrainResult, TrainSpec

logger = logging.getLogger(__name__)

# GR00T's tune flags - the model-tuning surface that lerobot does NOT have.
# Sensible default mirrors FinetuneConfig defaults (projector + diffusion on).
_DEFAULT_TUNE = {"llm": False, "visual": False, "projector": True, "diffusion": True}

_SUPPORTED_METHODS = {"full", "frozen_backbone", "expert_only"}


class Gr00tTrainer(Trainer):
    """Post-tune an NVIDIA GR00T N1.x policy via Isaac-GR00T launch_finetune.py.

    Args:
        groot_root: Path to the Isaac-GR00T checkout (contains
            ``gr00t/experiment/launch_finetune.py``). Falls back to the
            ``GR00T_ROOT`` env var, then ``TrainSpec.extra['groot_root']``.
        python_executable: Interpreter for the subprocess (default current).
    """

    def __init__(
        self,
        groot_root: str | None = None,
        python_executable: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.groot_root = groot_root or os.environ.get("GR00T_ROOT")
        self.python_executable = python_executable or sys.executable

    @property
    def provider_name(self) -> str:
        return "groot"

    @property
    def hardware_floor(self) -> dict[str, Any]:
        # N1.x fine-tune fits one modern GPU (lite); multi-GPU recommended.
        return {"min_gpus": 1, "min_vram_gb": 24, "multinode": False}

    # ---- helpers -----------------------------------------------------------

    def _resolve_groot_root(self, spec: TrainSpec) -> str | None:
        return self.groot_root or spec.extra.get("groot_root")

    def _launch_script(self, root: str) -> str:
        return os.path.join(root, "gr00t", "experiment", "launch_finetune.py")

    def _resolve_tune(self, spec: TrainSpec) -> dict[str, bool]:
        merged = dict(_DEFAULT_TUNE)
        merged.update({k: bool(v) for k, v in (spec.tune or {}).items()
                       if k in _DEFAULT_TUNE})
        # method=frozen_backbone => freeze llm+visual, keep projector/diffusion.
        if spec.method == "frozen_backbone":
            merged["llm"] = False
            merged["visual"] = False
        return merged

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
            problems.append("base_model is required (--base_model_path)")
        if not spec.output_dir:
            problems.append("output_dir is required")
        if not spec.embodiment:
            problems.append("embodiment is required for GR00T (--embodiment_tag)")

        if spec.method not in _SUPPORTED_METHODS:
            problems.append(
                f"unsupported method '{spec.method}' for GR00T "
                f"(expected one of {sorted(_SUPPORTED_METHODS)}); "
                f"use tune={{...}} for fine-grained control"
            )
        if spec.steps <= 0:
            problems.append(f"steps must be > 0, got {spec.steps}")

        root = self._resolve_groot_root(spec)
        if not root:
            problems.append(
                "Isaac-GR00T checkout not found; set GR00T_ROOT, pass "
                "groot_root=..., or extra['groot_root']"
            )
        elif not os.path.isfile(self._launch_script(root)):
            problems.append(
                f"launch_finetune.py not found under groot_root={root} "
                f"(expected {self._launch_script(root)})"
            )

        mcfg = spec.extra.get("modality_config_path")
        if mcfg and not os.path.isfile(mcfg):
            problems.append(f"modality_config_path does not exist: {mcfg}")

        return problems

    def build_command(self, spec: TrainSpec) -> list[str]:
        """Translate a TrainSpec into the launch_finetune.py argv (pure)."""
        root = self._resolve_groot_root(spec)
        script = self._launch_script(root) if root else "launch_finetune.py"

        if spec.num_gpus > 1:
            launcher = [
                "torchrun",
                f"--nproc_per_node={spec.num_gpus}",
                f"--master_port={spec.extra.get('master_port', 29500)}",
                script,
            ]
        else:
            launcher = [self.python_executable, script]

        cmd = [
            *launcher,
            f"--base_model_path={spec.base_model}",
            f"--dataset_path={spec.dataset_root}",
            f"--embodiment_tag={spec.embodiment}",
            f"--output_dir={spec.output_dir}",
            f"--max_steps={spec.steps}",
            f"--global_batch_size={spec.global_batch_size}",
            f"--learning_rate={spec.learning_rate}",
            f"--save_steps={spec.save_freq}",
            f"--num_gpus={spec.num_gpus}",
        ]

        # Tune flags (the GR00T-specific surface).
        tune = self._resolve_tune(spec)
        cmd.append(f"--tune_llm={'true' if tune['llm'] else 'false'}")
        cmd.append(f"--tune_visual={'true' if tune['visual'] else 'false'}")
        cmd.append(f"--tune_projector={'true' if tune['projector'] else 'false'}")
        cmd.append(f"--tune_diffusion_model={'true' if tune['diffusion'] else 'false'}")

        # Augmentation.
        if spec.augmentation:
            if "random_rotation_angle" in spec.augmentation:
                cmd.append(f"--random_rotation_angle={spec.augmentation['random_rotation_angle']}")
            if "color_jitter_params" in spec.augmentation:
                # tyro takes color_jitter as nested; pass as JSON via extra config instead
                cmd.append(
                    f"--extra_augmentation_config={json.dumps(spec.augmentation)}"
                )

        # Modality config (.py) registration.
        mcfg = spec.extra.get("modality_config_path")
        if mcfg:
            cmd.append(f"--modality_config_path={mcfg}")

        if spec.resume:
            cmd.append("--resume_from_checkpoint")

        # Passthrough: remaining extra.* as --key=value (skip consumed keys).
        _consumed = {"groot_root", "modality_config_path", "master_port"}
        for key, value in spec.extra.items():
            if key in _consumed:
                continue
            cmd.append(f"--{key}={value}")

        return cmd

    def train(self, spec: TrainSpec) -> TrainResult:
        problems = self.validate(spec)
        if problems:
            return TrainResult(
                status="error", job_id="",
                message="validation failed: " + "; ".join(problems),
            )

        self.prepare(spec)
        parent = os.path.dirname(os.path.abspath(spec.output_dir)) or "."
        os.makedirs(parent, exist_ok=True)

        cmd = self.build_command(spec)
        job_id = f"groot-{int(time.time())}"
        log_path = os.path.join(parent, f"{os.path.basename(spec.output_dir)}.{job_id}.log")

        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        # Single-GPU: pin one device so HF Trainer doesn't wrap in DataParallel
        # (the StopIteration crash documented in examples/finetune.sh).
        if spec.num_gpus <= 1:
            env.setdefault("CUDA_VISIBLE_DEVICES", env.get("CUDA_VISIBLE_DEVICES", "0"))

        logger.info("Gr00tTrainer launching: %s", " ".join(cmd))
        try:
            with open(log_path, "w", encoding="utf-8") as logf:
                proc = subprocess.run(  # noqa: S603 - argv values validated by validate()/_security_problems before train() builds the command; list form, no shell
                    cmd, cwd=parent, env=env,
                    stdout=logf, stderr=subprocess.STDOUT, check=False,
                )
        except FileNotFoundError as e:
            return TrainResult(
                status="error", job_id=job_id,
                message=f"launcher not found ({e}); is torchrun/python + Isaac-GR00T present?",
            )

        ckpt = self._latest_checkpoint(spec.output_dir)
        if proc.returncode != 0:
            return TrainResult(
                status="error", job_id=job_id, checkpoint_dir=ckpt,
                message=f"launch_finetune.py exited {proc.returncode}; see {log_path}",
            )
        return TrainResult(
            status="success", job_id=job_id, checkpoint_dir=ckpt,
            message=f"GR00T fine-tune complete; log: {log_path}",
        )

    def _latest_checkpoint(self, output_dir: str) -> str | None:
        """GR00T (HF Trainer) writes ``checkpoint-<step>`` dirs in output_dir."""
        if not os.path.isdir(output_dir):
            return None
        ckpts = [
            d for d in os.listdir(output_dir)
            if d.startswith("checkpoint-")
            and os.path.isdir(os.path.join(output_dir, d))
        ]
        if not ckpts:
            return None
        # Highest step number wins.
        def _step(name: str) -> int:
            try:
                return int(name.split("-", 1)[1])
            except (IndexError, ValueError):
                return -1
        best = max(ckpts, key=_step)
        return os.path.join(output_dir, best)
