"""GR00T trainer - drives Isaac-GR00T's finetune pipeline AS A LIBRARY.

GR00T N1.x ships its own post-training pipeline (NOT lerobot). Its
``gr00t/experiment/launch_finetune.py`` is only a thin ``__main__`` shim: it
parses a :class:`gr00t.configs.finetune_config.FinetuneConfig` via tyro,
translates it into a :class:`gr00t.configs.base_config.Config`, and calls
:func:`gr00t.experiment.experiment.run`. There is **no reusable function** in
that file - so to use GR00T as a library we reproduce that translation here
(building the very same objects the script builds) and call ``run(config)``
directly. No script file is re-executed, no ``sys.argv`` is parsed, no
``subprocess`` / ``torchrun`` binary is spawned.

* single GPU -> :func:`gr00t.experiment.experiment.run` called in THIS process.
* multi-GPU  -> torch's programmatic ``elastic_launch`` (the API behind
  ``torchrun``) spawns one worker per GPU; each builds the same ``Config`` and
  calls ``run`` - arguments are Python objects, never a command line.

Mapping (provider-agnostic ``TrainSpec`` -> GR00T ``FinetuneConfig`` fields):

* ``base_model``        -> ``base_model_path``
* ``dataset_root``      -> ``dataset_path``
* ``embodiment``        -> ``embodiment_tag`` (REQUIRED by GR00T)
* ``steps``             -> ``max_steps``
* ``global_batch_size`` -> ``global_batch_size``
* ``learning_rate``     -> ``learning_rate``
* ``save_freq``         -> ``save_steps``
* ``resume``            -> ``resume_from_checkpoint``
* ``tune`` dict         -> ``tune_llm/tune_visual/tune_projector/tune_diffusion_model``
* ``augmentation``      -> ``random_rotation_angle`` / ``color_jitter_params`` /
                           ``extra_augmentation_config``
* ``extra['modality_config_path']`` -> ``modality_config_path``
* any other ``extra[k]`` -> ``FinetuneConfig.k`` IF it is a real field (gated by
                            the dataclass fields; unknown keys ignored + warned).

GR00T checkpoints are HF-native, so :meth:`export` is the default passthrough.
The checkout is resolved from ``GR00T_ROOT`` / ``extra['groot_root']`` and put
on ``sys.path`` so ``import gr00t`` resolves the user's installed pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any

from strands_robots.training._inproc import call_callable, elastic_launch_callable
from strands_robots.training.base import Trainer, TrainResult, TrainSpec

logger = logging.getLogger(__name__)

# GR00T's tune flags - the model-tuning surface that lerobot does NOT have.
# Sensible default mirrors FinetuneConfig defaults (projector + diffusion on).
_DEFAULT_TUNE = {"llm": False, "visual": False, "projector": True, "diffusion": True}

_SUPPORTED_METHODS = {"full", "frozen_backbone", "expert_only"}

# FinetuneConfig fields consumed explicitly (not auto-passed from extra).
_CONSUMED_EXTRA = {"groot_root", "master_port"}


class Gr00tTrainer(Trainer):
    """Post-tune an NVIDIA GR00T N1.x policy via the Isaac-GR00T library.

    Args:
        groot_root: Path to the Isaac-GR00T checkout (the package root that
            contains ``gr00t/``). Falls back to the ``GR00T_ROOT`` env var, then
            ``TrainSpec.extra['groot_root']``. Added to ``sys.path`` so
            ``import gr00t`` resolves the user's pipeline.
    """

    def __init__(self, groot_root: str | None = None, **kwargs: Any) -> None:
        self.groot_root = groot_root or os.environ.get("GR00T_ROOT")

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

    def _ensure_importable(self, spec: TrainSpec) -> None:
        """Put the resolved checkout on sys.path so ``import gr00t`` works."""
        root = self._resolve_groot_root(spec)
        if root and root not in sys.path and os.path.isdir(os.path.join(root, "gr00t")):
            sys.path.insert(0, root)

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
            problems.append("base_model is required (FinetuneConfig.base_model_path)")
        if not spec.output_dir:
            problems.append("output_dir is required")
        if not spec.embodiment:
            problems.append("embodiment is required for GR00T (embodiment_tag)")

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
        elif not os.path.isdir(os.path.join(root, "gr00t")):
            problems.append(
                f"gr00t package not found under groot_root={root} "
                f"(expected {os.path.join(root, 'gr00t')})"
            )

        mcfg = spec.extra.get("modality_config_path")
        if mcfg and not os.path.isfile(mcfg):
            problems.append(f"modality_config_path does not exist: {mcfg}")

        return problems

    def build_finetune_config(self, spec: TrainSpec) -> Any:
        """Build GR00T's own ``FinetuneConfig`` object from a TrainSpec (pure).

        Returns an instance of ``gr00t.configs.finetune_config.FinetuneConfig``
        - the SAME typed object ``launch_finetune.py`` builds via tyro, but
        constructed directly from Python values (no argv, no shell). Requires
        the GR00T checkout importable; call :meth:`_ensure_importable` first.
        """
        from gr00t.configs.finetune_config import FinetuneConfig

        tune = self._resolve_tune(spec)
        kwargs: dict[str, Any] = {
            "base_model_path": spec.base_model,
            "dataset_path": spec.dataset_root,
            "embodiment_tag": spec.embodiment,
            "output_dir": spec.output_dir,
            "max_steps": spec.steps,
            "global_batch_size": spec.global_batch_size,
            "learning_rate": spec.learning_rate,
            "save_steps": spec.save_freq,
            "num_gpus": spec.num_gpus,
            "tune_llm": tune["llm"],
            "tune_visual": tune["visual"],
            "tune_projector": tune["projector"],
            "tune_diffusion_model": tune["diffusion"],
            "resume_from_checkpoint": spec.resume,
        }

        # Augmentation -> typed FinetuneConfig fields.
        if spec.augmentation:
            if "random_rotation_angle" in spec.augmentation:
                kwargs["random_rotation_angle"] = spec.augmentation["random_rotation_angle"]
            if "color_jitter_params" in spec.augmentation:
                kwargs["color_jitter_params"] = spec.augmentation["color_jitter_params"]
            # FinetuneConfig.extra_augmentation_config is a JSON STRING field.
            extra_aug = {
                k: v for k, v in spec.augmentation.items()
                if k not in ("random_rotation_angle", "color_jitter_params")
            }
            if extra_aug:
                kwargs["extra_augmentation_config"] = json.dumps(extra_aug)

        if spec.extra.get("modality_config_path"):
            kwargs["modality_config_path"] = spec.extra["modality_config_path"]

        # Passthrough: any other extra.* that is a REAL FinetuneConfig field.
        # We gate against the dataclass's own fields (typed allowlist) - an
        # unknown key can never set an attribute, let alone become a CLI flag.
        import dataclasses

        valid_fields = {f.name for f in dataclasses.fields(FinetuneConfig)}
        for key, value in spec.extra.items():
            if key in _CONSUMED_EXTRA or key in kwargs or key == "modality_config_path":
                continue
            if key in valid_fields:
                kwargs[key] = value
            else:
                logger.warning(
                    "Gr00tTrainer: ignoring extra '%s' (not a FinetuneConfig field).",
                    key,
                )

        return FinetuneConfig(**kwargs)

    def _build_run_config(self, ft_config: Any) -> Any:
        """Translate a ``FinetuneConfig`` into the ``Config`` ``run()`` consumes.

        Mirrors the body of ``launch_finetune.py``'s ``__main__`` exactly
        (verified against the Isaac-GR00T checkout) so calling ``run(config)``
        is behaviourally identical to launching the script - minus the process
        spawn and argv parse.
        """
        from gr00t.configs.base_config import get_default_config
        from gr00t.data.embodiment_tags import EmbodimentTag

        ft_config.embodiment_tag = EmbodimentTag.resolve(ft_config.embodiment_tag)
        embodiment_tag = ft_config.embodiment_tag.value

        # Register a user-provided modality config (.py) the same way the script does.
        if ft_config.modality_config_path is not None:
            self._load_modality_config(ft_config.modality_config_path)

        dataset_paths = [p for p in ft_config.dataset_path.split(os.pathsep) if p]

        config = get_default_config().load_dict(
            {
                "data": {
                    "download_cache": False,
                    "datasets": [
                        {
                            "dataset_paths": dataset_paths,
                            "mix_ratio": 1.0,
                            "embodiment_tag": embodiment_tag,
                        }
                    ],
                }
            }
        )
        config.load_config_path = None

        # Mirror the script's FinetuneConfig -> Config field copy.
        config.model.tune_llm = ft_config.tune_llm
        config.model.tune_visual = ft_config.tune_visual
        config.model.tune_projector = ft_config.tune_projector
        config.model.tune_diffusion_model = ft_config.tune_diffusion_model
        config.model.state_dropout_prob = ft_config.state_dropout_prob
        config.model.random_rotation_angle = ft_config.random_rotation_angle
        config.model.color_jitter_params = ft_config.color_jitter_params
        config.model.extra_augmentation_config = (
            json.loads(ft_config.extra_augmentation_config)
            if ft_config.extra_augmentation_config else None
        )
        config.model.load_bf16 = False
        config.model.reproject_vision = False
        config.model.model_name = "nvidia/Cosmos-Reason2-2B"
        config.model.backbone_trainable_params_fp32 = True
        config.model.use_relative_action = True

        config.training.experiment_name = ft_config.experiment_name
        config.training.start_from_checkpoint = ft_config.base_model_path
        config.training.optim = "adamw_torch"
        config.training.global_batch_size = ft_config.global_batch_size
        config.training.dataloader_num_workers = ft_config.dataloader_num_workers
        config.training.learning_rate = ft_config.learning_rate
        config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
        config.training.output_dir = ft_config.output_dir
        config.training.save_steps = ft_config.save_steps
        config.training.save_total_limit = ft_config.save_total_limit
        config.training.num_gpus = ft_config.num_gpus
        config.training.use_wandb = ft_config.use_wandb
        config.training.max_steps = ft_config.max_steps
        config.training.weight_decay = ft_config.weight_decay
        config.training.warmup_ratio = ft_config.warmup_ratio
        config.training.wandb_project = ft_config.wandb_project

        config.data.shard_size = ft_config.shard_size
        config.data.episode_sampling_rate = ft_config.episode_sampling_rate
        config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch

        config.training.save_only_model = ft_config.save_only_model
        config.training.resume_from_checkpoint = ft_config.resume_from_checkpoint
        config.training.skip_weight_loading = ft_config.skip_weight_loading

        return config

    @staticmethod
    def _load_modality_config(modality_config_path: str) -> None:
        """Register a user modality config (.py), mirroring launch_finetune.py."""
        import importlib
        from pathlib import Path

        path = Path(modality_config_path)
        if path.exists() and path.suffix == ".py":
            if str(path.parent) not in sys.path:
                sys.path.append(str(path.parent))
            importlib.import_module(path.stem)
            logger.info("Loaded modality config: %s", path)
        else:
            raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")

    def train(self, spec: TrainSpec) -> TrainResult:
        problems = self.validate(spec)
        if problems:
            return TrainResult(
                status="error", job_id="",
                message="validation failed: " + "; ".join(problems),
            )

        self.prepare(spec)
        self._ensure_importable(spec)
        parent = os.path.dirname(os.path.abspath(spec.output_dir)) or "."
        os.makedirs(parent, exist_ok=True)

        job_id = f"groot-{int(time.time())}"
        log_path = os.path.join(parent, f"{os.path.basename(spec.output_dir)}.{job_id}.log")

        # Single-GPU: pin one device so HF Trainer doesn't DataParallel-wrap
        # (the StopIteration crash documented in examples/finetune.sh).
        if spec.num_gpus <= 1:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
        os.environ.setdefault("LOGURU_LEVEL", "INFO")

        logger.info(
            "Gr00tTrainer launching GR00T run() in-process: num_gpus=%d steps=%d",
            spec.num_gpus, spec.steps,
        )

        train_error: BaseException | None = None
        try:
            if spec.num_gpus and spec.num_gpus > 1:
                # Multi-GPU: torch elastic agent spawns workers; each builds the
                # FinetuneConfig and calls GR00T run() - Python objects, no argv.
                groot_root = self._resolve_groot_root(spec)
                elastic_launch_callable(
                    _groot_worker,
                    nproc_per_node=spec.num_gpus,
                    nnodes=1,
                    run_id=job_id,
                    fn_args=(groot_root, spec, log_path),
                )
            else:
                ft_config = self.build_finetune_config(spec)
                run_config = self._build_run_config(ft_config)
                from gr00t.experiment.experiment import run
                call_callable(run, run_config, log_path=log_path)
        except BaseException as e:  # noqa: BLE001 - convert ANY failure to a result
            train_error = e
            logger.error("Gr00tTrainer in-process run failed: %s", e)

        ckpt = self._latest_checkpoint(spec.output_dir)
        if train_error is not None:
            return TrainResult(
                status="error", job_id=job_id, checkpoint_dir=ckpt,
                message=f"GR00T run() raised {type(train_error).__name__}: "
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
        def _step(name: str) -> int:
            try:
                return int(name.split("-", 1)[1])
            except (IndexError, ValueError):
                return -1
        best = max(ckpts, key=_step)
        return os.path.join(output_dir, best)


def _groot_worker(groot_root: str | None, spec: TrainSpec, log_path: str) -> None:
    """elastic_launch worker: build the GR00T Config and call run() in this worker.

    Runs in a torch-spawned worker process (one per GPU). torch sets RANK /
    LOCAL_RANK / WORLD_SIZE; GR00T's run() + HF Trainer read those to shard.
    We do NOT pin CUDA_VISIBLE_DEVICES here - each worker sees all devices and
    selects by LOCAL_RANK. Only local rank 0 tees to the shared log file.
    """
    if groot_root and groot_root not in sys.path and os.path.isdir(os.path.join(groot_root, "gr00t")):
        sys.path.insert(0, groot_root)
    trainer = Gr00tTrainer(groot_root=groot_root)
    ft_config = trainer.build_finetune_config(spec)
    run_config = trainer._build_run_config(ft_config)
    from gr00t.experiment.experiment import run

    is_rank0 = os.environ.get("LOCAL_RANK", "0") == "0"
    call_callable(run, run_config, log_path=log_path if is_rank0 else None)
