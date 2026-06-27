"""LeRobot trainer - drives ``lerobot.scripts.lerobot_train.train`` AS A LIBRARY.

Builds a typed :class:`lerobot.configs.train.TrainPipelineConfig` and calls
lerobot's ``train(cfg)`` **directly in this interpreter** for any LeRobot-native
policy type (act, diffusion, smolvla, pi0, pi05, ...). The training *logic* is
entirely lerobot's; this adapter only translates a provider-agnostic
:class:`~strands_robots.training.base.TrainSpec` into the config object, manages
resume, and parses the run for a status verdict.

Why in-process (no ``subprocess``)
----------------------------------
lerobot's entry point is a plain function ``train(cfg)`` whose ``@parser.wrap()``
decorator (lerobot ``configs/parser.py``) short-circuits when the first
positional arg is **already** a ``TrainPipelineConfig`` instance - it uses that
object verbatim and never reads ``sys.argv``. So we build the config as typed
Python objects (``make_policy_config`` + ``DatasetConfig`` + ``PeftConfig``) and
hand it straight to ``train(cfg)``. No shell, no argv, no second interpreter.

Launcher selection (still no shell):
    * 1 GPU / CPU    -> call ``train(cfg)`` directly in-process.
    * >1 GPU, 1 node -> ``elastic_launch`` (torch's programmatic launcher, the
      engine behind ``torchrun``); each worker builds the cfg and calls
      ``train(cfg)``. lerobot creates its own ``Accelerator`` inside, which picks
      up the worker's distributed env. No ``accelerate``/``torchrun`` binary.
    * multi-node     -> rejected in ``validate()`` (needs a per-node launcher).

:meth:`build_command` is retained as a PURE argv-parity helper - it documents
the exact draccus CLI the typed config corresponds to and powers the
``test_native_parity`` drift check. It is NOT used to launch anything.

Grounded against lerobot 0.5.x ``TrainPipelineConfig`` / ``DatasetConfig`` /
``PeftConfig``.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import shutil
import time
from typing import TYPE_CHECKING, Any

from strands_robots.training._inproc import call_callable, elastic_launch_callable
from strands_robots.training.base import Trainer, TrainResult, TrainSpec

if TYPE_CHECKING:  # pragma: no cover - typing only
    from lerobot.configs.train import TrainPipelineConfig

logger = logging.getLogger(__name__)

# LeRobot-native policy types (draccus --policy.type values / make_policy_config
# keys). Mirrors the verified vla-ft POLICY_MAP; values pass straight through.
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

# LeRobot policy types whose config exposes ``use_relative_actions`` (the
# relative-action processor pair: a RelativeActionsProcessorStep on the input
# side and the matching AbsoluteActionsProcessorStep on the output side, both
# built from ``config.use_relative_actions`` and saved into the checkpoint's
# pre/post processors). Other policy types have no such field.
_RELATIVE_ACTION_POLICY_TYPES = {"pi0", "pi05", "pi0_fast"}

# Hugging Face Hub dataset id: ``org/name`` (each segment alnum plus ._-). Used
# to gate the agent-supplied ``dataset_repo_id`` before it becomes lerobot's
# ``DatasetConfig.repo_id`` (which load_dataset/HfApi feed to a Hub URL).
_HUB_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


class LerobotTrainer(Trainer):
    """Post-tune a LeRobot-native policy by calling ``lerobot`` train in-process.

    Args:
        policy_type: LeRobot policy type (default ``"act"``). Resolved from
            ``TrainSpec.extra['policy_type']`` if present, else this.
        device: Torch device string (default auto: cuda > mps > cpu).
    """

    def __init__(
        self,
        policy_type: str = "act",
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.policy_type = policy_type
        self.device = device or _auto_device()

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

    def _resume_config_path(self, output_dir: str) -> str | None:
        """Return the resumable ``train_config.json`` FILE path, or None.

        lerobot writes checkpoints to ``<output_dir>/checkpoints/<step|last>/
        pretrained_model/train_config.json``; resume needs the FILE path (it
        derives policy_dir/checkpoint_path from it). This is the resume-wiring
        counterpart of the public :meth:`latest_checkpoint` (which returns the
        loadable DIRECTORY for ``export``/``create_policy``).
        """
        ckpts = os.path.join(output_dir, "checkpoints")
        if not os.path.isdir(ckpts):
            return None
        last = os.path.join(ckpts, "last", "pretrained_model", "train_config.json")
        if os.path.isfile(last):
            return last
        candidates = []
        for name in sorted(os.listdir(ckpts)):
            cfg = os.path.join(ckpts, name, "pretrained_model", "train_config.json")
            if os.path.isfile(cfg):
                candidates.append(cfg)
        return candidates[-1] if candidates else None

    def latest_checkpoint(self, output_dir: str) -> str | None:
        """Return the newest loadable ``pretrained_model`` directory, or None.

        ABC contract: a directory ``create_policy``/``export`` can consume.
        lerobot's loadable artifact is the ``pretrained_model`` dir that holds
        ``model.safetensors`` + ``train_config.json``; we locate it from the
        resume config file's parent.
        """
        cfg_file = self._resume_config_path(output_dir)
        return os.path.dirname(cfg_file) if cfg_file else None

    def _dataset_source(self, spec: TrainSpec) -> tuple[str, str | None]:
        """Resolve (repo_id, root) for lerobot's ``DatasetConfig``.

        Two mutually sufficient data sources, mirroring lerobot's own model:

        * Hub dataset (``spec.dataset_repo_id`` set) -> ``repo_id`` is the Hub
          id; ``root`` is the optional local cache dir (``spec.dataset_root`` or
          ``None``). With ``streaming=True`` lerobot streams shards from the Hub
          without a full download - the 50-500 GB disk-blowup fix.
        * Local v3 root (``spec.dataset_root`` only) -> ``repo_id="local"`` and
          ``root`` is the dataset path, unchanged from the record->train loop.
        """
        if spec.dataset_repo_id:
            return spec.dataset_repo_id, (spec.dataset_root or None)
        return "local", spec.dataset_root

    def _val_split_episodes(self, spec: TrainSpec) -> list[int] | None:
        """Held-out validation split: train on the FIRST (total - N) episodes.

        Requires a local ``meta/info.json`` to know the episode count, so it is
        a no-op for a Hub dataset with no local cache (``dataset_repo_id`` set,
        ``dataset_root`` empty) - the full Hub dataset is used in that case.
        """
        if spec.val_episodes is None:
            return None
        if not spec.dataset_root:
            return None
        total = self._dataset_total_episodes(spec.dataset_root)
        if total is not None and 0 < spec.val_episodes < total:
            return list(range(0, total - spec.val_episodes))
        return None

    def _relative_actions(self, spec: TrainSpec) -> bool:
        """Whether to train with relative (delta) actions (``extra['relative_actions']``).

        Relative-action training predicts deltas from the current robot state
        instead of absolute targets - part of the strongest manipulation
        ablations. lerobot implements it as a matched processor pair built from
        ``config.use_relative_actions``: a ``RelativeActionsProcessorStep`` on
        the input side (encode target->delta at train time) and the inverse
        ``AbsoluteActionsProcessorStep`` on the output side (decode delta->target
        at inference). Both are saved into the checkpoint's pre/post processors,
        so deployment via ``lerobot_local`` (which loads the saved processor
        pipeline) restores the inverse decode automatically - no separate
        inference-side wiring is needed.

        Only ``pi0`` / ``pi05`` / ``pi0_fast`` expose ``use_relative_actions``;
        :meth:`validate` rejects the flag for any other policy type rather than
        letting it become a silent no-op.
        """
        return bool(spec.extra.get("relative_actions", False))

    # ---- ABC ---------------------------------------------------------------

    def validate(self, spec: TrainSpec) -> list[str]:
        problems: list[str] = self._security_problems(spec)

        # Data source: either a local v3 root, or a Hub repo id (streaming the
        # 50-500 GB case without a full download). Exactly one must be present.
        if spec.dataset_repo_id:
            if not _HUB_REPO_ID_RE.match(spec.dataset_repo_id):
                problems.append(
                    f"dataset_repo_id '{spec.dataset_repo_id}' is not a valid Hub id "
                    "(expected 'org/name', alnum/._- segments)"
                )
            # dataset_root is optional here (local cache root); if given, it need
            # not yet contain meta/info.json (the Hub provides metadata).
        elif not spec.dataset_root:
            problems.append("a data source is required: set dataset_root (local v3) or dataset_repo_id (Hub)")
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
                f"policy_type '{ptype}' is not LeRobot-native (expected one of {sorted(_LEROBOT_POLICY_TYPES)})"
            )

        if spec.method not in _SUPPORTED_METHODS:
            problems.append(f"unsupported method '{spec.method}' (expected one of {sorted(_SUPPORTED_METHODS)})")
        if spec.method == "lora" and spec.tune.get("expert_only"):
            problems.append("lora and expert_only are mutually exclusive (both freeze the VLM)")

        if self._relative_actions(spec) and ptype not in _RELATIVE_ACTION_POLICY_TYPES:
            problems.append(
                f"relative_actions is not supported by policy_type '{ptype}' "
                f"(only {sorted(_RELATIVE_ACTION_POLICY_TYPES)} expose use_relative_actions); "
                "drop extra['relative_actions'] or pick a pi0-family policy"
            )

        if spec.steps <= 0:
            problems.append(f"steps must be > 0, got {spec.steps}")

        if spec.num_nodes > 1:
            problems.append(
                f"num_nodes={spec.num_nodes}: multi-node lerobot needs a per-node "
                "launcher and cannot run in-process; use num_nodes=1."
            )

        if spec.val_episodes is not None and spec.dataset_root:
            total = self._dataset_total_episodes(spec.dataset_root)
            if total is not None and spec.val_episodes >= total:
                problems.append(f"val_episodes={spec.val_episodes} >= total_episodes={total}")

        # lerobot must be importable to actually train.
        try:
            import importlib.util

            if importlib.util.find_spec("lerobot.scripts.lerobot_train") is None:
                problems.append("lerobot is not installed (no lerobot.scripts.lerobot_train)")
        except Exception:  # noqa: BLE001
            problems.append("lerobot is not installed")

        return problems

    def build_command(self, spec: TrainSpec) -> list[str]:
        """PURE argv-parity helper - the draccus CLI the typed config maps to.

        NOT used to launch training (``train()`` builds a typed config and calls
        lerobot's ``train(cfg)`` directly). Retained so ``test_native_parity``
        can assert our field mapping matches lerobot's real CLI, and as a
        human-readable description of the equivalent command.
        """
        ptype = self._resolve_policy_type(spec)
        repo_id, root = self._dataset_source(spec)
        cmd = [
            "lerobot.scripts.lerobot_train",
            f"--dataset.repo_id={repo_id}",
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
        if root:
            cmd.insert(2, f"--dataset.root={root}")
        if spec.streaming:
            cmd.append("--dataset.streaming=true")
        if spec.base_model:
            cmd.append(f"--policy.pretrained_path={spec.base_model}")
        if spec.seed is not None:
            cmd.append(f"--seed={spec.seed}")
        if spec.method == "lora":
            cmd.append("--peft.method_type=LORA")
            if spec.lora_r is not None:
                cmd.append(f"--peft.r={spec.lora_r}")
            if spec.lora_target_modules is not None:
                cmd.append(f"--peft.target_modules={spec.lora_target_modules}")
        elif spec.method == "expert_only":
            cmd.append("--policy.train_expert_only=true")
        if self._relative_actions(spec):
            cmd.append("--policy.use_relative_actions=true")
        eps = self._val_split_episodes(spec)
        if eps is not None:
            cmd.append(f"--dataset.episodes=[{', '.join(map(str, eps))}]")
        if spec.resume:
            ckpt_cfg = self._resume_config_path(spec.output_dir)
            if ckpt_cfg:
                cmd.append("--resume=true")
                cmd.append(f"--config_path={ckpt_cfg}")
        _consumed = {"policy_type", "job_name", "relative_actions"}
        for key, value in spec.extra.items():
            if key in _consumed:
                continue
            cmd.append(f"--{key}={value}")
        return cmd

    def build_config(self, spec: TrainSpec) -> TrainPipelineConfig:
        """Build lerobot's typed ``TrainPipelineConfig`` from a TrainSpec (pure).

        The in-process equivalent of :meth:`build_command`: constructs the
        dataclass tree ``train(cfg)`` consumes directly (no argv).
        """
        import dataclasses
        from pathlib import Path

        from lerobot.configs.default import DatasetConfig, PeftConfig
        from lerobot.configs.train import TrainPipelineConfig
        from lerobot.policies.factory import make_policy_config

        ptype = self._resolve_policy_type(spec)

        policy_cfg = make_policy_config(ptype)
        if hasattr(policy_cfg, "device"):
            policy_cfg.device = self.device
        if hasattr(policy_cfg, "push_to_hub"):
            policy_cfg.push_to_hub = False
        if spec.base_model:
            policy_cfg.pretrained_path = Path(spec.base_model)
        if spec.method == "expert_only" and hasattr(policy_cfg, "train_expert_only"):
            policy_cfg.train_expert_only = True
        if self._relative_actions(spec):
            if not hasattr(policy_cfg, "use_relative_actions"):
                raise ValueError(
                    f"relative_actions requested but policy_type '{ptype}' has no "
                    f"use_relative_actions field (supported: {sorted(_RELATIVE_ACTION_POLICY_TYPES)})"
                )
            policy_cfg.use_relative_actions = True

        repo_id, root = self._dataset_source(spec)
        dataset_kwargs: dict[str, Any] = {
            "repo_id": repo_id,
            "root": root,
            "episodes": self._val_split_episodes(spec),
        }
        if spec.streaming:
            dataset_kwargs["streaming"] = True
        dataset_cfg = DatasetConfig(**dataset_kwargs)

        peft_cfg = None
        if spec.method == "lora":
            peft_kwargs: dict[str, Any] = {"method_type": "LORA"}
            if spec.lora_r is not None:
                peft_kwargs["r"] = spec.lora_r
            if spec.lora_alpha is not None:
                peft_kwargs["lora_alpha"] = spec.lora_alpha
            if spec.lora_target_modules is not None:
                peft_kwargs["target_modules"] = spec.lora_target_modules
            supported = {f.name for f in dataclasses.fields(PeftConfig)}
            unsupported = sorted(k for k in peft_kwargs if k not in supported)
            if unsupported:
                raise ValueError(
                    f"The installed lerobot's PeftConfig does not support LoRA "
                    f"option(s) {unsupported}; it accepts {sorted(supported)}. "
                    "These options were requested via TrainSpec. Upgrade lerobot "
                    "to a version that supports them, or drop them from the spec."
                )
            peft_cfg = PeftConfig(**peft_kwargs)
            if hasattr(policy_cfg, "use_peft"):
                policy_cfg.use_peft = True

        cfg = TrainPipelineConfig(
            dataset=dataset_cfg,
            policy=policy_cfg,
            output_dir=Path(spec.output_dir) if spec.output_dir else None,
            job_name=str(spec.extra.get("job_name", "strands_ft")),
            steps=spec.steps,
            batch_size=spec.global_batch_size,
            save_freq=spec.save_freq,
            resume=spec.resume,
            peft=peft_cfg,
        )
        if spec.seed is not None:
            cfg.seed = spec.seed
        if hasattr(cfg, "wandb") and hasattr(cfg.wandb, "enable"):
            cfg.wandb.enable = False
        if spec.resume:
            ckpt_cfg = self._resume_config_path(spec.output_dir)
            if ckpt_cfg:
                cfg.checkpoint_path = Path(ckpt_cfg).parent.parent

        # Typed passthrough for remaining extra.* (gated by validate()'s key
        # allowlist). Only set attributes that exist on the typed config tree;
        # unknown keys are ignored (never become an arbitrary flag).
        _consumed = {"policy_type", "job_name", "relative_actions"}
        for key, value in spec.extra.items():
            if key in _consumed:
                continue
            target, attr = _resolve_dotted(cfg, key)
            if target is not None and hasattr(target, attr):
                setattr(target, attr, value)
            else:
                logger.warning("LerobotTrainer: ignoring extra '%s' (no matching config field).", key)
        return cfg

    def train(self, spec: TrainSpec) -> TrainResult:
        problems = self.validate(spec)
        if problems:
            return TrainResult(
                status="error",
                job_id="",
                message="validation failed: " + "; ".join(problems),
            )

        self.prepare(spec)

        # lerobot's validate() REFUSES a pre-existing output_dir unless
        # resume=True. Don't pre-create output_dir; write our log NEXT TO it.
        parent = os.path.dirname(os.path.abspath(spec.output_dir)) or "."
        os.makedirs(parent, exist_ok=True)

        # Fresh-start hygiene: clear a stale output_dir with no resumable ckpt.
        if not spec.resume and os.path.isdir(spec.output_dir):
            if self.latest_checkpoint(spec.output_dir) is None:
                shutil.rmtree(spec.output_dir, ignore_errors=True)

        job_id = f"lerobot-{int(time.time())}"
        log_path = os.path.join(parent, f"{os.path.basename(spec.output_dir)}.{job_id}.log")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        try:
            cfg = self.build_config(spec)
        except Exception as e:  # noqa: BLE001 - config build is the typed boundary
            return TrainResult(
                status="error",
                job_id=job_id,
                message=f"failed to build lerobot TrainPipelineConfig: {e}",
            )

        logger.info(
            "LerobotTrainer launching in-process: policy=%s device=%s steps=%d num_gpus=%d",
            self._resolve_policy_type(spec),
            self.device,
            spec.steps,
            spec.num_gpus,
        )

        train_error: Exception | None = None
        try:
            if spec.num_gpus and spec.num_gpus > 1:
                elastic_launch_callable(
                    _lerobot_worker,
                    nproc_per_node=spec.num_gpus,
                    nnodes=1,
                    run_id=job_id,
                    fn_args=(self.policy_type, self.device, spec, log_path),
                )
            else:
                from lerobot.scripts.lerobot_train import train as lerobot_train

                call_callable(lerobot_train, cfg, log_path=log_path)
        except Exception as e:  # noqa: BLE001 - convert ANY failure to a result
            train_error = e
            logger.error("LerobotTrainer in-process train failed: %s", e)

        ckpt_model_dir = self.latest_checkpoint(spec.output_dir)  # loadable pretrained_model dir
        metrics = self._parse_log(log_path)

        if train_error is not None:
            return TrainResult(
                status="error",
                job_id=job_id,
                checkpoint_dir=ckpt_model_dir,
                metrics=metrics,
                message=f"lerobot train raised {type(train_error).__name__}: {train_error}; see {log_path}",
            )

        return TrainResult(
            status="success",
            job_id=job_id,
            checkpoint_dir=ckpt_model_dir,
            metrics=metrics,
            message=f"lerobot train complete (in-process); log: {log_path}",
        )

    def _parse_log(self, log_path: str) -> dict[str, Any]:
        """Extract a 'RUNNING != learning' verdict from the captured train log.

        Parses lerobot's MetricsTracker line (verified vs lerobot 0.5.x
        ``utils/logging_utils.py::MetricsTracker.__str__``)::

            step:1.2K smpl:4.9K ep:8 epch:2.00 loss:0.123 ...
        """
        latest_step: int | None = None
        latest_loss: float | None = None
        latest_epoch: float | None = None
        try:
            with open(log_path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if "step:" not in line:
                        continue
                    for tok in line.split():
                        key, _, val = tok.partition(":")
                        if not val:
                            continue
                        if key == "step":
                            n = _expand_big_number(val)
                            if n is not None:
                                latest_step = int(n)
                        elif key == "loss":
                            with contextlib.suppress(ValueError):
                                latest_loss = float(val)
                        elif key == "epch":
                            with contextlib.suppress(ValueError):
                                latest_epoch = float(val)
        except OSError:
            return {}

        metrics: dict[str, Any] = {}
        if latest_step is not None:
            metrics["latest_step"] = latest_step
        if latest_epoch is not None:
            metrics["latest_epoch"] = latest_epoch
        if latest_loss is not None:
            import math

            metrics["latest_loss"] = latest_loss
            metrics["learning"] = math.isfinite(latest_loss)
        metrics["liveness_ok"] = latest_step is not None
        return metrics


def _resolve_dotted(cfg: Any, key: str) -> tuple[Any, str]:
    """Map a (optionally dotted) extra key to (obj, attr) on the config tree."""
    if "." not in key:
        return cfg, key
    head, _, tail = key.partition(".")
    sub = getattr(cfg, head, None)
    if sub is None or "." in tail:
        return None, tail
    return sub, tail


def _lerobot_worker(policy_type: str, device: str, spec: TrainSpec, log_path: str) -> None:
    """elastic_launch worker: build the cfg and call lerobot train() in this worker.

    Runs in a torch-spawned worker (one per GPU). torch sets RANK / LOCAL_RANK /
    WORLD_SIZE; lerobot's Accelerator picks them up. Only local rank 0 tees to
    the shared log to avoid interleaved writes.
    """
    import os as _os

    trainer = LerobotTrainer(policy_type=policy_type, device=device)
    cfg = trainer.build_config(spec)
    from lerobot.scripts.lerobot_train import train as lerobot_train

    is_rank0 = _os.environ.get("LOCAL_RANK", "0") == "0"
    call_callable(lerobot_train, cfg, log_path=log_path if is_rank0 else None)


def _expand_big_number(token: str) -> float | None:
    """Invert lerobot's ``format_big_number`` (e.g. ``"1.2K" -> 1200``)."""
    suffixes = {"": 1, "K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12, "Q": 1e15}
    token = token.strip()
    if not token:
        return None
    suffix = token[-1].upper()
    if suffix in suffixes and suffix != "" and not token[-1].isdigit():
        body, mult = token[:-1], suffixes[suffix]
    else:
        body, mult = token, 1
    try:
        return float(body) * mult
    except ValueError:
        return None


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
