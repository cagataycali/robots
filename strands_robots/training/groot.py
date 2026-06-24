"""GR00T trainer - in-process wrapper over Isaac-GR00T's ``launch_finetune.py``.

GR00T N1.7 ships its own post-training pipeline (NOT lerobot): a
``FinetuneConfig`` dataclass driven via tyro through
``gr00t/experiment/launch_finetune.py``. This adapter translates a
provider-agnostic :class:`~strands_robots.training.base.TrainSpec` into that
script's argument list and runs it **without spawning a shell**:

* single GPU -> the script is executed IN-PROCESS via :func:`runpy.run_path`
  with a controlled ``sys.argv`` list (``CUDA_VISIBLE_DEVICES=0`` pins one
  device so HF Trainer doesn't DataParallel-wrap, per ``examples/finetune.sh``).
* multi-GPU  -> torch's programmatic ``elastic_launch`` (the API behind
  ``torchrun``) spawns one worker per GPU and each runs the script in-process;
  no ``torchrun`` binary, no command line.

Why not ``subprocess`` + ``torchrun``
-------------------------------------
The previous implementation built a string ``argv`` partly from
caller-controlled ``TrainSpec.extra`` (each ``extra[k]=v`` -> ``--k=v``) and
handed it to ``subprocess.run`` / the ``torchrun`` executable. Assembling a
command line from external input for a spawned interpreter is a needless
injection / arbitrary-flag surface. We now build an argv **list** (never a
shell string), gate every passthrough key through
:func:`~strands_robots.training._inproc.safe_flag_key`, and execute via
:mod:`runpy` / ``elastic_launch`` so external input is never interpreted by a
shell.

Mapping highlights (unchanged from the CLI semantics):

* ``base_model``        -> ``--base_model_path``
* ``dataset_root``      -> ``--dataset_path``
* ``embodiment``        -> ``--embodiment_tag`` (REQUIRED by GR00T)
* ``steps``             -> ``--max_steps``
* ``global_batch_size`` -> ``--global_batch_size``
* ``learning_rate``     -> ``--learning_rate``
* ``save_freq``         -> ``--save_steps``
* ``resume``            -> ``--resume_from_checkpoint``
* ``tune`` dict         -> ``--tune_llm/--tune_visual/--tune_projector/--tune_diffusion_model``
* ``augmentation``      -> ``--random_rotation_angle`` / ``--extra_augmentation_config`` (JSON)
* ``extra['modality_config_path']`` -> ``--modality_config_path``

GR00T checkpoints are HF-native, so :meth:`export` is the default passthrough.
The script path is resolved from the ``GR00T_ROOT`` env var (the Isaac-GR00T
checkout) or ``extra['groot_root']``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from strands_robots.training._inproc import (
    elastic_launch_callable,
    filter_safe_extra,
    run_python_path,
)
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
        python_executable: Deprecated / ignored. Kept so existing callers
            constructing ``Gr00tTrainer(python_executable=...)`` don't break;
            the script now runs in THIS interpreter (or torch-spawned workers),
            so there is no child process to point at a different Python.
    """

    def __init__(
        self,
        groot_root: str | None = None,
        python_executable: str | None = None,  # noqa: ARG002 - back-compat shim, ignored
        **kwargs: Any,
    ) -> None:
        self.groot_root = groot_root or os.environ.get("GR00T_ROOT")
        if python_executable is not None:
            logger.debug(
                "Gr00tTrainer(python_executable=%r) is ignored: launch_finetune.py "
                "now runs in-process (no subprocess).",
                python_executable,
            )

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
        problems: list[str] = []

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

        if spec.num_nodes > 1:
            problems.append(
                f"num_nodes={spec.num_nodes}: multi-node GR00T needs a per-node "
                "rendezvous; run one in-process Gr00tTrainer per node with a shared "
                "rdzv endpoint, or use num_nodes=1."
            )

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

    def build_args(self, spec: TrainSpec) -> list[str]:
        """Translate a TrainSpec into the launch_finetune.py argument LIST (pure).

        Returns ONLY the script's flags - no launcher binary, no script path.
        The caller prepends the script path (single-GPU) or hands this list to a
        worker that does the same under ``elastic_launch`` (multi-GPU). Building a
        list (not a shell string) plus the safe-key gate is what removes the old
        injection surface.
        """
        args = [
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
        args.append(f"--tune_llm={'true' if tune['llm'] else 'false'}")
        args.append(f"--tune_visual={'true' if tune['visual'] else 'false'}")
        args.append(f"--tune_projector={'true' if tune['projector'] else 'false'}")
        args.append(f"--tune_diffusion_model={'true' if tune['diffusion'] else 'false'}")

        # Augmentation.
        if spec.augmentation:
            if "random_rotation_angle" in spec.augmentation:
                args.append(f"--random_rotation_angle={spec.augmentation['random_rotation_angle']}")
            if "color_jitter_params" in spec.augmentation:
                # tyro takes color_jitter as nested; pass as JSON via extra config instead
                args.append(
                    f"--extra_augmentation_config={json.dumps(spec.augmentation)}"
                )

        # Modality config (.py) registration.
        mcfg = spec.extra.get("modality_config_path")
        if mcfg:
            args.append(f"--modality_config_path={mcfg}")

        if spec.resume:
            args.append("--resume_from_checkpoint")

        # Passthrough: remaining extra.* as --key=value, but ONLY keys that pass
        # the safe-key gate (no spaces / shell metacharacters / leading dashes).
        # Unsafe keys are dropped with a warning - they can never become a token.
        _consumed = {"groot_root", "modality_config_path", "master_port"}
        safe, rejected = filter_safe_extra(spec.extra, _consumed)
        for key, value in safe.items():
            args.append(f"--{key}={value}")
        for key in rejected:
            logger.warning(
                "Gr00tTrainer: ignoring unsafe extra key %r (not [A-Za-z0-9_.-]+).",
                key,
            )

        return args

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

        root = self._resolve_groot_root(spec)
        script = self._launch_script(root)
        args = self.build_args(spec)
        job_id = f"groot-{int(time.time())}"
        log_path = os.path.join(parent, f"{os.path.basename(spec.output_dir)}.{job_id}.log")

        # Env hints (set for the run, restored after by run_python_path).
        env = {"PYTHONUNBUFFERED": "1"}
        # Single-GPU: pin one device so HF Trainer doesn't wrap in DataParallel
        # (the StopIteration crash documented in examples/finetune.sh).
        if spec.num_gpus <= 1:
            env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

        logger.info(
            "Gr00tTrainer launching in-process: script=%s num_gpus=%d steps=%d",
            script, spec.num_gpus, spec.steps,
        )

        train_error: BaseException | None = None
        try:
            if spec.num_gpus and spec.num_gpus > 1:
                # Multi-GPU: torch elastic agent spawns workers; each runs the
                # script in-process with the SAME argv list (Python objects, no
                # command line). torchrun-equivalent without the binary.
                elastic_launch_callable(
                    _groot_worker,
                    nproc_per_node=spec.num_gpus,
                    nnodes=1,
                    run_id=job_id,
                    fn_args=(script, args, parent, log_path),
                )
            else:
                run_python_path(
                    script, args, cwd=parent, env=env, log_path=log_path,
                )
        except BaseException as e:  # noqa: BLE001 - convert ANY failure to a result
            train_error = e
            logger.error("Gr00tTrainer in-process launch failed: %s", e)

        ckpt = self._latest_checkpoint(spec.output_dir)
        if train_error is not None:
            return TrainResult(
                status="error", job_id=job_id, checkpoint_dir=ckpt,
                message=f"launch_finetune.py raised {type(train_error).__name__}: "
                        f"{train_error}; see {log_path}",
            )
        return TrainResult(
            status="success", job_id=job_id, checkpoint_dir=ckpt,
            message=f"GR00T fine-tune complete (in-process); log: {log_path}",
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


def _groot_worker(script: str, args: list[str], cwd: str, log_path: str) -> None:
    """elastic_launch worker entry point: run launch_finetune.py in this worker.

    Runs in a torch-spawned worker process (one per GPU). torch sets RANK /
    LOCAL_RANK / WORLD_SIZE in the environment; the GR00T script + HF Trainer
    read those to shard. We do NOT pin CUDA_VISIBLE_DEVICES here - each worker
    must see all devices and select by LOCAL_RANK (HF Trainer's DDP path).
    Only the local rank 0 worker tees to the shared log file to avoid interleave.
    """
    is_rank0 = os.environ.get("LOCAL_RANK", "0") == "0"
    run_python_path(
        script, args, cwd=cwd, env={"PYTHONUNBUFFERED": "1"},
        log_path=log_path if is_rank0 else None,
    )
