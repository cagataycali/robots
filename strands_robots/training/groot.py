"""GR00T trainer — wrapper over Isaac-GR00T's ``launch_finetune``.

GR00T N1.7 ships its own post-training pipeline (NOT lerobot): a
``FinetuneConfig`` dataclass driven via tyro through
``gr00t.experiment.launch_finetune``. We drive it as a Python library
(``importlib.import_module(...).main(argv)``) — same interpreter, no nested
``python`` invocation. Multi-GPU goes through :mod:`torch.distributed.run`
(the same engine ``torchrun`` is), also invoked in-process — mirroring
``examples/finetune.sh`` semantically without shell calls.

This adapter translates a provider-agnostic
:class:`~strands_robots.training.base.TrainSpec` into the ``launch_finetune``
argv. Mapping highlights:

* ``base_model``        → ``--base_model_path``
* ``dataset_root``      → ``--dataset_path``
* ``embodiment``        → ``--embodiment_tag`` (REQUIRED by GR00T)
* ``steps``             → ``--max_steps``
* ``global_batch_size`` → ``--global_batch_size``
* ``learning_rate``     → ``--learning_rate``
* ``save_freq``         → ``--save_steps``
* ``resume``            → ``--resume_from_checkpoint``
* ``tune`` dict         → ``--tune_llm/--tune_visual/--tune_projector/--tune_diffusion_model``
* ``augmentation``      → ``--random_rotation_angle`` / ``--color_jitter_params``
                          / ``--extra_augmentation_config`` (JSON)
* ``extra['modality_config_path']`` → ``--modality_config_path``

GR00T checkpoints are HF-native, so :meth:`export` is the default passthrough.
The Isaac-GR00T checkout is resolved from the ``GR00T_ROOT`` env var or
``extra['groot_root']`` — needed so we can ``chdir`` there for relative configs.
Install Isaac-GR00T from source (per its README) and ensure ``gr00t`` is
importable from the active interpreter; this trainer drives it as a Python
library, NOT by invoking another interpreter.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import json
import logging
import os
import time
from typing import Any

from strands_robots.training.base import Trainer, TrainResult, TrainSpec

logger = logging.getLogger(__name__)

# GR00T's tune flags — the model-tuning surface lerobot does NOT have.
# Sensible default mirrors FinetuneConfig defaults (projector + diffusion on).
_DEFAULT_TUNE = {"llm": False, "visual": False, "projector": True, "diffusion": True}

_SUPPORTED_METHODS = {"full", "frozen_backbone", "expert_only"}

_INSTALL_HINT = (
    "Isaac-GR00T is not importable from this interpreter. Install it from source "
    "(see https://github.com/NVIDIA/Isaac-GR00T or the path passed via "
    "groot_root / GR00T_ROOT) into the *same* Python that imports strands_robots "
    "— e.g. `pip install -e $GR00T_ROOT`."
)


def _import_groot_module(qualname: str) -> Any:
    """Import ``gr00t.<qualname>`` or raise a helpful ImportError."""
    full = f"gr00t.{qualname}"
    try:
        return importlib.import_module(full)
    except ImportError as e:  # pragma: no cover — exercised in integration
        raise ImportError(f"{_INSTALL_HINT} (failed to import {full})") from e


class Gr00tTrainer(Trainer):
    """Post-tune an NVIDIA GR00T N1.x policy via Isaac-GR00T launch_finetune.

    Args:
        groot_root: Path to the Isaac-GR00T checkout (used to ``chdir`` so
            relative configs/datasets resolve, and as the validation target
            for ``launch_finetune.py``). Falls back to the ``GR00T_ROOT`` env
            var, then ``TrainSpec.extra['groot_root']``. The package itself
            is loaded via :func:`importlib.import_module` from the active
            interpreter — install from source; ``GR00T_ROOT`` is for runtime
            config resolution, not the interpreter path.
    """

    def __init__(
        self,
        groot_root: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.groot_root = groot_root or os.environ.get("GR00T_ROOT")

    @property
    def provider_name(self) -> str:
        return "groot"

    @property
    def hardware_floor(self) -> dict[str, Any]:
        # N1.x fine-tune fits one modern GPU (lite); multi-GPU recommended.
        return {"min_gpus": 1, "min_vram_gb": 24, "multinode": False}

    def _resolve_groot_root(self, spec: TrainSpec) -> str | None:
        return self.groot_root or spec.extra.get("groot_root")

    def _launch_script(self, root: str) -> str:
        return os.path.join(root, "gr00t", "experiment", "launch_finetune.py")

    def _resolve_tune(self, spec: TrainSpec) -> dict[str, bool]:
        merged = dict(_DEFAULT_TUNE)
        merged.update({k: bool(v) for k, v in (spec.tune or {}).items() if k in _DEFAULT_TUNE})
        # method=frozen_backbone => freeze llm+visual, keep projector/diffusion.
        if spec.method == "frozen_backbone":
            merged["llm"] = False
            merged["visual"] = False
        return merged

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
                "Isaac-GR00T checkout not found; set GR00T_ROOT, pass groot_root=..., or extra['groot_root']"
            )
        elif not os.path.isfile(self._launch_script(root)):
            problems.append(
                f"launch_finetune.py not found under groot_root={root} (expected {self._launch_script(root)})"
            )

        mcfg = spec.extra.get("modality_config_path")
        if mcfg and not os.path.isfile(mcfg):
            problems.append(f"modality_config_path does not exist: {mcfg}")

        return problems

    def build_command(self, spec: TrainSpec) -> list[str]:
        """Translate a TrainSpec into the launch_finetune argv (pure).

        Returned as a faithful argv list so the parity tests can assert flag
        drift against the real ``FinetuneConfig``; at run time we hand the
        same tail (the script + flags) to :mod:`torch.distributed.run` /
        ``gr00t.experiment.launch_finetune.main`` in-process.
        """
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
            launcher = ["python", script]

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

        tune = self._resolve_tune(spec)
        cmd.append(f"--tune_llm={'true' if tune['llm'] else 'false'}")
        cmd.append(f"--tune_visual={'true' if tune['visual'] else 'false'}")
        cmd.append(f"--tune_projector={'true' if tune['projector'] else 'false'}")
        cmd.append(f"--tune_diffusion_model={'true' if tune['diffusion'] else 'false'}")

        if spec.augmentation:
            if "random_rotation_angle" in spec.augmentation:
                cmd.append(f"--random_rotation_angle={spec.augmentation['random_rotation_angle']}")
            if "color_jitter_params" in spec.augmentation:
                cmd.append(f"--extra_augmentation_config={json.dumps(spec.augmentation)}")

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

    def _launch_argv(self, spec: TrainSpec) -> list[str]:
        """Argv passed directly to ``launch_finetune.main`` (script arg stripped)."""
        # ``build_command`` prepends either ['torchrun', ..., script] (multi-GPU)
        # or ['python', script] (single-GPU); strip the launcher + script tokens
        # to leave just the flags ``launch_finetune.main`` parses.
        full = self.build_command(spec)
        # Find the script token (ends with launch_finetune.py).
        for i, tok in enumerate(full):
            if tok.endswith("launch_finetune.py"):
                return list(full[i + 1 :])
        # Fallback: assume single-GPU layout [python, script, *flags].
        return list(full[2:])

    def train(self, spec: TrainSpec) -> TrainResult:
        problems = self.validate(spec)
        if problems:
            return TrainResult(
                status="error",
                job_id="",
                message="validation failed: " + "; ".join(problems),
            )

        self.prepare(spec)
        parent = os.path.dirname(os.path.abspath(spec.output_dir)) or "."
        os.makedirs(parent, exist_ok=True)

        job_id = f"groot-{int(time.time())}"
        log_path = os.path.join(parent, f"{os.path.basename(spec.output_dir)}.{job_id}.log")
        root = self._resolve_groot_root(spec)

        # Single-GPU: pin one device so HF Trainer doesn't wrap in DataParallel
        # (the StopIteration crash documented in examples/finetune.sh).
        prev_env: dict[str, str | None] = {
            "PYTHONUNBUFFERED": os.environ.get("PYTHONUNBUFFERED"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        os.environ["PYTHONUNBUFFERED"] = "1"
        if spec.num_gpus <= 1:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

        argv = self._launch_argv(spec)
        prev_cwd = os.getcwd()
        rc = 0
        try:
            if root and os.path.isdir(root):
                os.chdir(root)

            with (
                open(log_path, "w", encoding="utf-8") as logf,
                contextlib.redirect_stdout(logf),
                contextlib.redirect_stderr(logf),
            ):
                if spec.num_gpus > 1:
                    # Multi-GPU: drive torch.distributed.run in-process — the same
                    # engine torchrun uses, no subprocess for the launcher itself.
                    from torch.distributed import run as torch_run

                    distributed_argv = [
                        f"--nproc_per_node={spec.num_gpus}",
                        f"--master_port={spec.extra.get('master_port', 29500)}",
                        self._launch_script(root) if root else "launch_finetune.py",
                        *argv,
                    ]
                    try:
                        torch_run.main(distributed_argv)
                    except SystemExit as se:
                        rc = int(se.code) if se.code is not None else 0
                else:
                    # Single-GPU: import and call gr00t.experiment.launch_finetune
                    # directly — no subprocess, no nested interpreter.
                    mod = _import_groot_module("experiment.launch_finetune")
                    main = getattr(mod, "main", None)
                    if main is None:
                        # Some Isaac-GR00T revisions name it differently; fall
                        # back to running the module's __main__ block.
                        spec_obj = importlib.util.find_spec("gr00t.experiment.launch_finetune")
                        if spec_obj is None:
                            raise ImportError(
                                "gr00t.experiment.launch_finetune is not importable; "
                                "is Isaac-GR00T installed in this interpreter?"
                            )
                        runpy = importlib.import_module("runpy")
                        try:
                            runpy.run_module(
                                "gr00t.experiment.launch_finetune",
                                run_name="__main__",
                                alter_sys=True,
                            )
                        except SystemExit as se:
                            rc = int(se.code) if se.code is not None else 0
                    else:
                        try:
                            # tyro-driven entrypoints typically read sys.argv;
                            # rewrite it for the duration of the call.
                            import sys as _sys

                            _saved = _sys.argv
                            _sys.argv = ["launch_finetune", *argv]
                            try:
                                main()
                            finally:
                                _sys.argv = _saved
                        except SystemExit as se:
                            rc = int(se.code) if se.code is not None else 0
        except ImportError as e:
            return TrainResult(
                status="error",
                job_id=job_id,
                message=str(e),
            )
        except Exception as e:  # noqa: BLE001 — surface launcher errors
            return TrainResult(
                status="error",
                job_id=job_id,
                message=f"launch_finetune raised: {e}; see {log_path}",
            )
        finally:
            os.chdir(prev_cwd)
            for k, v in prev_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        ckpt = self._latest_checkpoint(spec.output_dir)
        if rc != 0:
            return TrainResult(
                status="error",
                job_id=job_id,
                checkpoint_dir=ckpt,
                message=f"launch_finetune exited {rc}; see {log_path}",
            )
        return TrainResult(
            status="success",
            job_id=job_id,
            checkpoint_dir=ckpt,
            message=f"GR00T fine-tune complete; log: {log_path}",
        )

    def _latest_checkpoint(self, output_dir: str) -> str | None:
        """GR00T (HF Trainer) writes ``checkpoint-<step>`` dirs in output_dir."""
        if not os.path.isdir(output_dir):
            return None
        ckpts = [
            d
            for d in os.listdir(output_dir)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
        ]
        if not ckpts:
            return None

        def _step(name: str) -> int:
            try:
                return int(name.split("-", 1)[1])
            except (IndexError, ValueError):
                return -1

        best = max(ckpts, key=_step)
        return os.path.join(output_dir, best)
