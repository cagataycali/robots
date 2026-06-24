"""LeRobot trainer - in-process wrapper over ``lerobot.scripts.lerobot_train.train``.

Builds a typed :class:`~lerobot.configs.train.TrainPipelineConfig` and calls
lerobot's ``train(cfg)`` **directly in this interpreter** for any LeRobot-native
policy type (act, diffusion, smolvla, pi0, pi05, ...). The training *logic* is
entirely lerobot's; this adapter only translates a provider-agnostic
:class:`~strands_robots.training.base.TrainSpec` into the right config object,
manages resume, and parses the run for a status verdict.

Why in-process (no ``subprocess``)
----------------------------------
The previous implementation shelled out to ``python -m lerobot.scripts.lerobot_train``
with a string ``argv`` assembled partly from the caller-controlled ``TrainSpec.extra``
dict (each ``extra[k]=v`` became a ``--k=v`` token). Spawning an interpreter on a
command line built from external input is a needless injection / arbitrary-flag
surface. lerobot's entry point is a plain function:

    @parser.wrap()
    def train(cfg: TrainPipelineConfig, accelerator=None): ...

and its ``@parser.wrap()`` decorator (lerobot ``configs/parser.py``) short-circuits
when the first positional arg is **already** a ``TrainPipelineConfig`` instance -
it then uses that object verbatim and never touches ``sys.argv``. So we build the
config as typed Python objects (``make_policy_config`` + ``DatasetConfig`` +
``PeftConfig``) and hand it straight to ``train(cfg)``. No shell, no argv, no
string interpolation of caller input.

Launcher selection (still no shell):
    * 1 GPU / CPU      -> call ``train(cfg)`` directly in-process.
    * >1 GPU, 1 node   -> ``accelerate.notebook_launcher(train, (cfg,),
                          num_processes=N)`` - spawns workers via multiprocessing,
                          not a subprocess command line.
    * multi-node       -> genuinely needs a per-node process launcher
                          (``torchrun``/``accelerate launch``); we surface a clear
                          error rather than silently degrading.

Grounded against lerobot 0.5.x ``TrainPipelineConfig`` / ``DatasetConfig`` /
``PeftConfig`` (the dataclasses ``train()`` consumes).
"""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from typing import TYPE_CHECKING, Any

from strands_robots.training.base import Trainer, TrainResult, TrainSpec

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids importing lerobot at module load
    from lerobot.configs.train import TrainPipelineConfig

logger = logging.getLogger(__name__)

# LeRobot-native policy types (the ``make_policy_config`` keys). Mirrors the
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
    """Post-tune a LeRobot-native policy by calling ``lerobot`` train in-process.

    Args:
        policy_type: LeRobot policy type (default ``"act"``). Resolved from
            ``TrainSpec.extra['policy_type']`` if present, else this.
        device: Torch device string (default auto: cuda > mps > cpu).
        python_executable: Deprecated / ignored. Kept only so existing callers
            constructing ``LerobotTrainer(python_executable=...)`` don't break;
            training now runs in THIS interpreter, so there is no child process
            to point at a different Python.
    """

    def __init__(
        self,
        policy_type: str = "act",
        device: str | None = None,
        python_executable: str | None = None,  # noqa: ARG002 - back-compat shim, ignored
        **kwargs: Any,
    ) -> None:
        self.policy_type = policy_type
        self.device = device or _auto_device()
        if python_executable is not None:
            logger.debug(
                "LerobotTrainer(python_executable=%r) is ignored: training now runs "
                "in-process (no subprocess).",
                python_executable,
            )

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
        pretrained_model/train_config.json`` and resume needs the FILE path
        (it derives policy_dir/checkpoint_path from it).
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

    def _val_split_episodes(self, spec: TrainSpec) -> list[int] | None:
        """Held-out validation split: train on the FIRST (total - N) episodes."""
        if spec.val_episodes is None:
            return None
        total = self._dataset_total_episodes(spec.dataset_root)
        if total is not None and 0 < spec.val_episodes < total:
            return list(range(0, total - spec.val_episodes))
        return None

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

        if spec.num_nodes > 1:
            problems.append(
                f"num_nodes={spec.num_nodes}: multi-node training needs a per-node "
                "process launcher (torchrun/accelerate launch) and cannot run "
                "in-process. Launch one in-process LerobotTrainer per node under "
                "your own torchrun, or use num_nodes=1."
            )

        # lerobot must be importable to actually train.
        try:
            import importlib.util

            if importlib.util.find_spec("lerobot.scripts.lerobot_train") is None:
                problems.append("lerobot is not installed (no lerobot.scripts.lerobot_train)")
        except Exception:  # noqa: BLE001
            problems.append("lerobot is not installed")

        return problems

    def build_config(self, spec: TrainSpec) -> TrainPipelineConfig:
        """Translate a TrainSpec into a typed ``TrainPipelineConfig`` (pure, testable).

        This is the in-process replacement for the old ``build_command``: instead
        of emitting an argv list of ``--flag=value`` strings, it constructs the
        dataclass tree lerobot's ``train(cfg)`` consumes directly. No shell, no
        string interpolation of caller-controlled ``extra``.
        """
        from lerobot.configs.default import DatasetConfig, PeftConfig
        from lerobot.configs.train import TrainPipelineConfig
        from lerobot.policies.factory import make_policy_config

        ptype = self._resolve_policy_type(spec)

        # --- policy config (typed; == draccus --policy.type=<ptype>) ---------
        policy_cfg = make_policy_config(ptype)
        # device + don't push to hub from an unattended trainer.
        if hasattr(policy_cfg, "device"):
            policy_cfg.device = self.device
        if hasattr(policy_cfg, "push_to_hub"):
            policy_cfg.push_to_hub = False
        # Post-tune FROM a pretrained checkpoint when given (== --policy.pretrained_path).
        if spec.base_model:
            from pathlib import Path

            policy_cfg.pretrained_path = Path(spec.base_model)
        # expert-only freezes the VLM (pi0/smolvla expose this knob).
        if spec.method == "expert_only" and hasattr(policy_cfg, "train_expert_only"):
            policy_cfg.train_expert_only = True

        # --- dataset config (== --dataset.repo_id/root/episodes) -------------
        dataset_cfg = DatasetConfig(
            repo_id="local",
            root=spec.dataset_root,
            episodes=self._val_split_episodes(spec),
        )

        # --- LoRA / PEFT (== --peft.*) ---------------------------------------
        peft_cfg = None
        if spec.method == "lora":
            peft_kwargs: dict[str, Any] = {"method_type": "LORA"}
            if spec.lora_r is not None:
                peft_kwargs["r"] = spec.lora_r
            if spec.lora_alpha is not None:
                peft_kwargs["lora_alpha"] = spec.lora_alpha
            if spec.lora_target_modules is not None:
                # PeftConfig.target_modules accepts a str (suffix/regex/'all-linear')
                # or a list; pass the spec value through unchanged.
                peft_kwargs["target_modules"] = spec.lora_target_modules
            peft_cfg = PeftConfig(**peft_kwargs)
            if hasattr(policy_cfg, "use_peft"):
                policy_cfg.use_peft = True

        # --- top-level training config (== --output_dir/steps/batch_size/...) -
        from pathlib import Path

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
        # wandb off for unattended runs.
        if hasattr(cfg, "wandb") and hasattr(cfg.wandb, "enable"):
            cfg.wandb.enable = False

        # Resume: point at the latest checkpoint's train_config.json so lerobot
        # rehydrates the run (matches CLI --resume=true --config_path=<ckpt>).
        if spec.resume:
            ckpt_cfg = self._latest_checkpoint(spec.output_dir)
            if ckpt_cfg:
                cfg.checkpoint_path = Path(ckpt_cfg).parent.parent

        # --- typed passthrough for remaining extra.* (NO shell) --------------
        # Only set attributes that actually exist on the config dataclasses, and
        # only via setattr on the typed object - never as an argv string. Unknown
        # keys are ignored with a warning (same tolerance rule as Policy kwargs).
        _consumed = {"policy_type", "job_name"}
        for key, value in spec.extra.items():
            if key in _consumed:
                continue
            target, attr = self._resolve_extra_target(cfg, key)
            if target is not None and hasattr(target, attr):
                setattr(target, attr, value)
            else:
                logger.warning(
                    "LerobotTrainer: ignoring unknown extra '%s' (no matching "
                    "TrainPipelineConfig field).",
                    key,
                )

        return cfg

    @staticmethod
    def _resolve_extra_target(cfg: TrainPipelineConfig, key: str):
        """Map a dotted extra key (e.g. 'dataset.num_workers') to (obj, attr).

        A bare key targets the top-level config. A dotted key walks the typed
        sub-configs (``dataset``, ``policy``, ``wandb``, ...). Returns
        ``(None, key)`` when the path can't be resolved.
        """
        if "." not in key:
            return cfg, key
        head, _, tail = key.partition(".")
        sub = getattr(cfg, head, None)
        if sub is None:
            return None, tail
        # support one level of nesting (sufficient for lerobot's config tree)
        if "." in tail:
            return None, tail
        return sub, tail

    def train(self, spec: TrainSpec) -> TrainResult:
        problems = self.validate(spec)
        if problems:
            return TrainResult(
                status="error", job_id="",
                message="validation failed: " + "; ".join(problems),
            )

        self.prepare(spec)

        # lerobot's validate() REFUSES a pre-existing output_dir unless
        # resume=True. So we must NOT pre-create output_dir. We only ensure its
        # PARENT exists and write our capture log NEXT TO output_dir.
        parent = os.path.dirname(os.path.abspath(spec.output_dir)) or "."
        os.makedirs(parent, exist_ok=True)

        # Fresh-start hygiene: if NOT resuming and output_dir exists with no
        # resumable checkpoint, clear it so lerobot's guard doesn't crash a rerun.
        if not spec.resume and os.path.isdir(spec.output_dir):
            if self._latest_checkpoint(spec.output_dir) is None:
                shutil.rmtree(spec.output_dir, ignore_errors=True)

        job_id = f"lerobot-{int(time.time())}"
        log_path = os.path.join(parent, f"{os.path.basename(spec.output_dir)}.{job_id}.log")

        # Resume-friendly env (matches the old subprocess env hints).
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        try:
            cfg = self.build_config(spec)
        except Exception as e:  # noqa: BLE001 - config build is the typed boundary
            return TrainResult(
                status="error", job_id=job_id,
                message=f"failed to build lerobot TrainPipelineConfig: {e}",
            )

        logger.info(
            "LerobotTrainer launching in-process: policy=%s device=%s steps=%d num_gpus=%d",
            self._resolve_policy_type(spec), self.device, spec.steps, spec.num_gpus,
        )

        # Capture lerobot's logging + stdout/stderr to a file so _parse_log can
        # still produce the "RUNNING != learning" verdict (lerobot logs its
        # MetricsTracker line via the root logger / tqdm).
        capture = _LogCapture(log_path)
        train_error: BaseException | None = None
        try:
            with capture:
                self._invoke_train(cfg, spec)
        except BaseException as e:  # noqa: BLE001 - we convert ANY failure to a result
            train_error = e
            logger.error("LerobotTrainer in-process train failed: %s", e)

        ckpt_dir = self._latest_checkpoint(spec.output_dir)
        ckpt_model_dir = os.path.dirname(ckpt_dir) if ckpt_dir else None  # .../pretrained_model
        metrics = self._parse_log(log_path)

        if train_error is not None:
            return TrainResult(
                status="error", job_id=job_id,
                checkpoint_dir=ckpt_model_dir, metrics=metrics,
                message=f"lerobot train raised {type(train_error).__name__}: "
                        f"{train_error}; see {log_path}",
            )

        return TrainResult(
            status="success", job_id=job_id,
            checkpoint_dir=ckpt_model_dir, metrics=metrics,
            message=f"lerobot train complete (in-process); log: {log_path}",
        )

    def _invoke_train(self, cfg: TrainPipelineConfig, spec: TrainSpec) -> None:
        """Call lerobot's train(cfg) in-process; multiprocessing for multi-GPU.

        Single GPU/CPU -> direct call (zero new processes). Multi-GPU single node
        -> accelerate.notebook_launcher, which uses multiprocessing (NOT a shell
        command line) to spawn one worker per GPU. Each worker calls train(cfg);
        lerobot creates its own Accelerator inside, which picks up the launcher's
        distributed env. ``train`` is ``@parser.wrap()``-decorated but
        short-circuits to use the passed-in cfg verbatim (never reads sys.argv).
        """
        from lerobot.scripts.lerobot_train import train
        from lerobot.utils.import_utils import register_third_party_plugins

        # main() does this before train(); preserve plugin registration so
        # third-party policy types resolve identically to the CLI path.
        register_third_party_plugins()

        if spec.num_gpus and spec.num_gpus > 1:
            from accelerate import notebook_launcher

            notebook_launcher(train, (cfg,), num_processes=spec.num_gpus)
        else:
            train(cfg)

    def _parse_log(self, log_path: str) -> dict[str, Any]:
        """Extract a 'RUNNING != learning' verdict from the captured train log.

        Parses lerobot's MetricsTracker line, whose exact format (verified vs
        lerobot 0.5.x ``utils/logging_utils.py::MetricsTracker.__str__``) is::

            step:1.2K smpl:4.9K ep:8 epch:2.00 loss:0.123 ...

        - ``step`` / ``smpl`` / ``ep`` are run through ``format_big_number`` so
          they carry K/M/B/T/Q suffixes (``step:1.2K``); we expand them back.
        - ``loss`` is the AverageMeter avg, formatted ``:.3f``.

        We take the LAST occurrence of each (newest). ``learning`` is True when
        we saw a finite loss; ``liveness_ok`` is True when we saw a step line at
        all. Best-effort; returns ``{}`` if the log is unreadable.
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
                            try:
                                latest_loss = float(val)
                            except ValueError:
                                pass
                        elif key == "epch":
                            try:
                                latest_epoch = float(val)
                            except ValueError:
                                pass
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

    def status(self, job_id: str) -> TrainResult:
        """In-process training is synchronous, so there is no detached job to poll.

        ``train()`` returns only after the run finishes (or raises), with metrics
        already parsed. This stub exists for ABC parity / future detached runs.
        """
        return TrainResult(
            status="error", job_id=job_id,
            message="lerobot_local runs training synchronously in-process; "
                    "the TrainResult from train() already carries the verdict.",
        )


class _LogCapture:
    """Context manager: tee lerobot's logging + stdout/stderr into a file.

    lerobot's ``train`` configures the root logger (``init_logging``) and prints
    its MetricsTracker line; tqdm writes to stderr. We attach a FileHandler to the
    root logger and redirect stdout/stderr to the same file for the duration of
    the run, so ``_parse_log`` can read the verdict exactly as it did from the
    old subprocess log. The handler is always removed on exit.
    """

    def __init__(self, log_path: str) -> None:
        self.log_path = log_path
        self._fh: logging.FileHandler | None = None
        self._stream: io.TextIOBase | None = None
        self._redirect_out = None
        self._redirect_err = None

    def __enter__(self) -> _LogCapture:
        self._stream = open(self.log_path, "w", encoding="utf-8")  # noqa: SIM115
        self._fh = logging.FileHandler(self.log_path)
        self._fh.setLevel(logging.INFO)
        self._fh.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger().addHandler(self._fh)
        # Tee stdout/stderr too (MetricsTracker line is often print()/tqdm).
        self._redirect_out = redirect_stdout(_Tee(sys.stdout, self._stream))
        self._redirect_err = redirect_stderr(_Tee(sys.stderr, self._stream))
        self._redirect_out.__enter__()
        self._redirect_err.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        try:
            if self._redirect_err is not None:
                self._redirect_err.__exit__(*exc)
            if self._redirect_out is not None:
                self._redirect_out.__exit__(*exc)
        finally:
            root = logging.getLogger()
            if self._fh is not None:
                root.removeHandler(self._fh)
                self._fh.close()
            if self._stream is not None:
                self._stream.close()


class _Tee(io.TextIOBase):
    """Write-through tee: forwards writes to both a live console and a log file."""

    def __init__(self, primary: Any, secondary: Any) -> None:
        self._primary = primary
        self._secondary = secondary

    def write(self, s: str) -> int:  # type: ignore[override]
        try:
            self._primary.write(s)
        except Exception:  # noqa: BLE001 - never let logging break training
            pass
        try:
            self._secondary.write(s)
        except Exception:  # noqa: BLE001
            pass
        return len(s)

    def flush(self) -> None:  # type: ignore[override]
        for s in (self._primary, self._secondary):
            try:
                s.flush()
            except Exception:  # noqa: BLE001
                pass


def _expand_big_number(token: str) -> float | None:
    """Invert lerobot's ``format_big_number`` (e.g. ``"1.2K" -> 1200``).

    Suffixes (lerobot ``utils.py``): "" K M B T Q (powers of 1000). Returns the
    numeric value, or None if the token isn't a recognised big-number string.
    """
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
