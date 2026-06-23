"""LeRobot trainer - thin wrapper over ``lerobot.scripts.lerobot_train``.

Builds and launches the LeRobot training CLI for any LeRobot-native policy
type (act, diffusion, smolvla, pi0, pi05, ...). The training *logic* is
entirely lerobot's; this adapter only translates a provider-agnostic
:class:`~strands_robots.training.base.TrainSpec` into the correct draccus
command + launcher, manages resume, and parses the run for a status verdict.

Grounded against lerobot 0.5.x ``TrainPipelineConfig`` (the draccus config
``lerobot_train`` parses):
    --dataset.repo_id / --dataset.root / --dataset.episodes
    --policy.type / --policy.device / --policy.push_to_hub / --policy.pretrained_path
    --output_dir / --job_name / --steps / --batch_size / --save_freq
    --resume / --seed / --wandb.enable
    --peft.method_type / --peft.r / --peft.target_modules   (LoRA)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from typing import Any

from strands_robots.training.base import Trainer, TrainResult, TrainSpec

logger = logging.getLogger(__name__)

# LeRobot-native policy types (draccus --policy.type values). Mirrors the
# verified vla-ft POLICY_MAP; values pass straight through to lerobot.
_LEROBOT_POLICY_TYPES = {
    "act",
    "diffusion",
    "vqbet",
    "tdmpc",
    "smolvla",
    "pi0",
    "pi05",
    "pi0_fast",
    "groot",
    "xvla",
}

_SUPPORTED_METHODS = {"full", "lora", "expert_only"}


class LerobotTrainer(Trainer):
    """Post-tune a LeRobot-native policy via ``lerobot.scripts.lerobot_train``.

    Args:
        policy_type: LeRobot ``--policy.type`` (default ``"act"``). Resolved
            from ``TrainSpec.extra['policy_type']`` if present, else this.
        device: ``--policy.device`` (default auto: cuda > mps > cpu).
        python_executable: Interpreter for the subprocess (default current).
    """

    def __init__(
        self,
        policy_type: str = "act",
        device: str | None = None,
        python_executable: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.policy_type = policy_type
        self.device = device or _auto_device()
        self.python_executable = python_executable or sys.executable

    @property
    def provider_name(self) -> str:
        return "lerobot_local"

    @property
    def hardware_floor(self) -> dict[str, Any]:
        # ACT fits a consumer GPU; large VLAs (pi05) want an L40S. Advisory.
        return {"min_gpus": 1, "min_vram_gb": 8, "multinode": False}

    # ---- helpers -----------------------------------------------------------

    def _resolve_policy_type(self, spec: TrainSpec) -> str:
        return str(spec.extra.get("policy_type", self.policy_type))

    def _dataset_total_episodes(self, dataset_root: str) -> int | None:
        info = os.path.join(dataset_root, "meta", "info.json")
        try:
            with open(info, encoding="utf-8") as f:
                return int(json.load(f).get("total_episodes"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    def _latest_checkpoint(self, output_dir: str) -> str | None:
        """Return the resumable ``train_config.json`` path, or None.

        lerobot writes checkpoints to ``<output_dir>/checkpoints/<step|last>/
        pretrained_model/train_config.json`` and validate() needs the FILE
        path on resume (it derives policy_dir/checkpoint_path from it).
        """
        ckpts = os.path.join(output_dir, "checkpoints")
        if not os.path.isdir(ckpts):
            return None
        # Prefer 'last', else the highest-numbered checkpoint dir.
        candidates = []
        last = os.path.join(ckpts, "last", "pretrained_model", "train_config.json")
        if os.path.isfile(last):
            return last
        for name in sorted(os.listdir(ckpts)):
            cfg = os.path.join(ckpts, name, "pretrained_model", "train_config.json")
            if os.path.isfile(cfg):
                candidates.append(cfg)
        return candidates[-1] if candidates else None

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

        if not spec.output_dir:
            problems.append("output_dir is required")

        ptype = self._resolve_policy_type(spec)
        if ptype not in _LEROBOT_POLICY_TYPES:
            problems.append(
                f"policy_type '{ptype}' is not LeRobot-native "
                f"(expected one of {sorted(_LEROBOT_POLICY_TYPES)})"
            )

        if spec.method not in _SUPPORTED_METHODS:
            problems.append(
                f"unsupported method '{spec.method}' "
                f"(expected one of {sorted(_SUPPORTED_METHODS)})"
            )
        if spec.method == "lora" and spec.tune.get("expert_only"):
            problems.append("lora and expert_only are mutually exclusive (both freeze the VLM)")

        if spec.steps <= 0:
            problems.append(f"steps must be > 0, got {spec.steps}")

        if spec.val_episodes is not None and spec.dataset_root:
            total = self._dataset_total_episodes(spec.dataset_root)
            if total is not None and spec.val_episodes >= total:
                problems.append(
                    f"val_episodes={spec.val_episodes} >= total_episodes={total}"
                )

        # lerobot must be importable to actually train.
        try:
            import importlib.util

            if importlib.util.find_spec("lerobot.scripts.lerobot_train") is None:
                problems.append("lerobot is not installed (no lerobot.scripts.lerobot_train)")
        except Exception:  # noqa: BLE001
            problems.append("lerobot is not installed")

        return problems

    def build_command(self, spec: TrainSpec) -> list[str]:
        """Translate a TrainSpec into the lerobot_train argv (pure, testable)."""
        ptype = self._resolve_policy_type(spec)

        if spec.num_gpus > 1:
            launcher = [
                "accelerate", "launch",
                "--multi_gpu",
                f"--num_processes={spec.num_gpus}",
                "--num_machines=1",
                "--mixed_precision=bf16",
                "-m", "lerobot.scripts.lerobot_train",
            ]
        else:
            launcher = [self.python_executable, "-m", "lerobot.scripts.lerobot_train"]

        cmd = [
            *launcher,
            "--dataset.repo_id=local",
            f"--dataset.root={spec.dataset_root}",
            f"--policy.type={ptype}",
            f"--policy.device={self.device}",
            "--policy.push_to_hub=false",
            f"--output_dir={spec.output_dir}",
            f"--job_name={spec.extra.get('job_name', 'strands_ft')}",
            f"--steps={spec.steps}",
            f"--batch_size={spec.global_batch_size}",
            f"--save_freq={spec.save_freq}",
            "--wandb.enable=false",
        ]

        if spec.base_model:
            cmd.append(f"--policy.pretrained_path={spec.base_model}")
        if spec.seed is not None:
            cmd.append(f"--seed={spec.seed}")

        # Tuning strategy.
        if spec.method == "lora":
            cmd.append("--peft.method_type=LORA")
            if spec.lora_r is not None:
                cmd.append(f"--peft.r={spec.lora_r}")
            if spec.lora_target_modules is not None:
                cmd.append(f"--peft.target_modules={spec.lora_target_modules}")
        elif spec.method == "expert_only":
            cmd.append("--policy.train_expert_only=true")

        # Held-out validation split: train on the FIRST (total - N) episodes.
        if spec.val_episodes is not None:
            total = self._dataset_total_episodes(spec.dataset_root)
            if total is not None and 0 < spec.val_episodes < total:
                train_eps = list(range(0, total - spec.val_episodes))
                cmd.append(f"--dataset.episodes=[{', '.join(map(str, train_eps))}]")

        # Resume: needs BOTH --resume=true AND --config_path=<ckpt>/train_config.json
        if spec.resume:
            ckpt_cfg = self._latest_checkpoint(spec.output_dir)
            if ckpt_cfg:
                cmd.append("--resume=true")
                cmd.append(f"--config_path={ckpt_cfg}")

        # Passthrough: any remaining extra.* as --key=value (skip consumed keys).
        _consumed = {"policy_type", "job_name"}
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

        # lerobot's validate() REFUSES a pre-existing output_dir unless
        # resume=True (it creates the dir itself). So we must NOT pre-create
        # output_dir. We only ensure its PARENT exists, and we write our log
        # NEXT TO output_dir (not inside it) so the log file doesn't trip the
        # "already exists" guard either.
        parent = os.path.dirname(os.path.abspath(spec.output_dir)) or "."
        os.makedirs(parent, exist_ok=True)

        # Fresh-start hygiene: if NOT resuming and output_dir exists with no
        # resumable checkpoint, clear it so lerobot's guard doesn't crash a
        # rerun (the vla-ft reclaim-before-first-save failure mode).
        if not spec.resume and os.path.isdir(spec.output_dir):
            if self._latest_checkpoint(spec.output_dir) is None:
                shutil.rmtree(spec.output_dir, ignore_errors=True)

        cmd = self.build_command(spec)
        job_id = f"lerobot-{int(time.time())}"
        log_path = os.path.join(parent, f"{os.path.basename(spec.output_dir)}.{job_id}.log")

        logger.info("LerobotTrainer launching: %s", " ".join(cmd))
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        try:
            with open(log_path, "w", encoding="utf-8") as logf:
                proc = subprocess.run(  # noqa: S603 - cmd is built from a vetted allowlist
                    cmd, cwd=parent, env=env,
                    stdout=logf, stderr=subprocess.STDOUT, check=False,
                )
        except FileNotFoundError as e:
            return TrainResult(
                status="error", job_id=job_id,
                message=f"launcher not found ({e}); is lerobot/accelerate installed?",
            )

        ckpt_dir = self._latest_checkpoint(spec.output_dir)
        ckpt_model_dir = (
            os.path.dirname(ckpt_dir) if ckpt_dir else None  # .../pretrained_model
        )
        metrics = self._parse_log(log_path)

        if proc.returncode != 0:
            return TrainResult(
                status="error", job_id=job_id,
                checkpoint_dir=ckpt_model_dir, metrics=metrics,
                message=f"lerobot_train exited {proc.returncode}; see {log_path}",
            )

        return TrainResult(
            status="success", job_id=job_id,
            checkpoint_dir=ckpt_model_dir, metrics=metrics,
            message=f"lerobot_train complete; log: {log_path}",
        )

    def _parse_log(self, log_path: str) -> dict[str, Any]:
        """Extract a 'RUNNING != learning' verdict from the train log tail.

        lerobot logs lines like ``... step:200 ... loss:0.123 ...``. We grab the
        last step/loss we can find. Best-effort; returns {} if unreadable.
        """
        latest_step: int | None = None
        latest_loss: float | None = None
        try:
            with open(log_path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    for tok in line.replace(",", " ").split():
                        if tok.startswith("step:"):
                            try:
                                latest_step = int(tok.split(":", 1)[1])
                            except ValueError:
                                pass
                        elif tok.startswith("loss:"):
                            try:
                                latest_loss = float(tok.split(":", 1)[1])
                            except ValueError:
                                pass
        except OSError:
            return {}
        metrics: dict[str, Any] = {}
        if latest_step is not None:
            metrics["latest_step"] = latest_step
        if latest_loss is not None:
            metrics["latest_loss"] = latest_loss
            metrics["learning"] = latest_loss == latest_loss  # not NaN
        metrics["liveness_ok"] = latest_step is not None
        return metrics


def _auto_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"
