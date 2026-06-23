"""Trainer abstraction - post-tune ANY policy provider natively.

The :class:`Trainer` ABC is the training-side peer of
:class:`~strands_robots.policies.base.Policy` (inference). Where ``Policy``
hides *how a model produces actions*, ``Trainer`` hides *how a model is
post-tuned* - and those pipelines genuinely differ per provider:

* **LeRobot** - ``python -m lerobot.scripts.lerobot_train`` (draccus CLI),
  ``accelerate launch`` for multi-GPU. HF-native checkpoints.
* **GR00T N1.7** - ``gr00t/experiment/launch_finetune.py`` (a ``FinetuneConfig``
  dataclass via tyro), ``torchrun`` for multi-GPU, ``tune_llm/visual/projector/
  diffusion`` flags + a modality-config ``.py``.
* **Cosmos3** - ``torchrun -m cosmos_framework.scripts.train --sft-toml=...``
  (TOML recipe + Hydra overrides), an explicit **DCP checkpoint conversion**
  prepare step, and a **DCP -> safetensors** export step. 8xH100 floor.

All three nonetheless converge on:

1. the same **dataset format** - LeRobotDataset v3 (what
   :class:`~strands_robots.dataset_recorder.DatasetRecorder` already writes), and
2. the same **lifecycle** - ``validate -> prepare -> train -> export``.

A ``Trainer`` is selected by the SAME provider name as its ``Policy``
(``groot`` / ``lerobot_local`` / ``cosmos3``), so a single registry identity
owns both the inference class and the training class. Adding a new policy =
add a ``Policy`` + a ``Trainer`` under one provider entry.

See :class:`MockTrainer` (``strands_robots/training/mock.py``) for the canonical
no-dependency reference implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainSpec:
    """Provider-agnostic post-tuning specification.

    Concrete trainers read the fields they support and **ignore the rest** -
    the same tolerance rule that :meth:`Policy.get_actions` applies to its
    ``**kwargs``. Backends MUST NOT raise on a field they don't use; new,
    backend-specific knobs live in :attr:`extra` until >=2 backends share
    them and they graduate to a first-class field.

    Attributes:
        dataset_root: Path to a LeRobotDataset v3 root (must contain
            ``meta/info.json``). This is exactly what
            :class:`~strands_robots.dataset_recorder.DatasetRecorder` /
            ``Robot.stop_recording`` produce, so the ``record -> train`` loop
            needs no conversion layer.
        base_model: HF model id or local checkpoint path to post-tune *from*.
        output_dir: Directory for checkpoints, logs, and the final artifact.
        embodiment: Embodiment tag / robot id. Required by GR00T
            (``--embodiment_tag``); LeRobot infers it from dataset features;
            optional elsewhere.
        steps: Total optimizer steps (maps to lerobot ``--steps`` /
            GR00T ``max_steps`` / Cosmos ``trainer.max_iter``).
        global_batch_size: Batch summed across GPUs before grad accumulation.
        learning_rate: Initial LR.
        save_freq: Checkpoint cadence in steps.
        num_gpus: GPUs on this node. ``>1`` selects the multi-GPU launcher
            (``accelerate`` for lerobot, ``torchrun`` for groot/cosmos).
        num_nodes: Nodes for multi-node training (Cosmos HSDP /
            ``torchrun --nnodes``).
        resume: Resume from the latest checkpoint under ``output_dir`` when
            one exists.
        seed: Master seed (best-effort; not all backends expose it).
        method: Tuning strategy, mapped per-backend:
            ``"full"`` | ``"lora"`` | ``"expert_only"`` | ``"frozen_backbone"``.
            ``lora`` and ``expert_only`` are mutually exclusive (both freeze
            the VLM); a backend MUST reject the combination in
            :meth:`Trainer.validate`.
        lora_r / lora_alpha / lora_target_modules: LoRA hyperparameters
            (used only when ``method == "lora"``). ``lora_target_modules=None``
            means "use the policy's built-in default targets".
        tune: Fine-grained component toggles for backends that expose them
            (GR00T: ``{"llm": bool, "visual": bool, "projector": bool,
            "diffusion": bool}``). Ignored by backends that don't.
        val_episodes: Hold out the LAST N episodes as a validation set
            (deterministic split; lerobot ``--dataset.episodes=[0..total-N-1]``).
        augmentation: Backend-specific data augmentation (GR00T
            ``color_jitter_params`` / ``random_rotation_angle``; Cosmos
            dataset filter dict).
        fps: Dataset control rate, when a backend needs it explicitly.
        extra: Raw passthrough. Keys become backend-native flags / overrides
            (lerobot ``--key=value``; Cosmos Hydra ``key.path=value``). The
            escape hatch that keeps the ABC stable as backends evolve.
    """

    # --- universal ---
    dataset_root: str
    base_model: str
    output_dir: str
    embodiment: str | None = None
    steps: int = 10_000
    global_batch_size: int = 32
    learning_rate: float = 1e-4
    save_freq: int = 1_000
    num_gpus: int = 1
    num_nodes: int = 1
    resume: bool = False
    seed: int | None = None
    # --- tuning strategy ---
    method: str = "full"
    lora_r: int | None = None
    lora_alpha: int | None = None
    lora_target_modules: str | None = None
    tune: dict[str, bool] = field(default_factory=dict)
    # --- data ---
    val_episodes: int | None = None
    augmentation: dict[str, Any] | None = None
    fps: int | None = None
    # --- escape hatch ---
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainResult:
    """Outcome of a training lifecycle call.

    Attributes:
        status: ``"success"`` | ``"running"`` | ``"error"``.
        job_id: Stable id for this run (used by :meth:`Trainer.status`).
        checkpoint_dir: Where checkpoints are written (``None`` before any
            save / on validation failure).
        exported_model: Final loadable artifact path - a value that
            ``create_policy(...)`` can consume - once :meth:`Trainer.export`
            has run. ``None`` otherwise.
        metrics: Free-form metrics for the "RUNNING != learning" verdict
            (e.g. ``latest_step``, ``latest_loss``, ``learning``,
            ``liveness_ok``).
        message: Human-readable status / error detail.
    """

    status: str
    job_id: str
    checkpoint_dir: str | None = None
    exported_model: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    message: str = ""


class Trainer(ABC):
    """Abstract base class for post-tuning a policy of one provider family.

    Lifecycle: :meth:`validate` (pure preflight) -> :meth:`prepare` (optional
    one-time setup) -> :meth:`train` (launch + manage) -> :meth:`export`
    (produce a loadable artifact). :meth:`status` answers
    "is it *really* learning?" for an in-flight job.

    Concrete trainers are thin adapters that translate a :class:`TrainSpec`
    into their backend's native launch (a subprocess command, a config object,
    a TOML recipe) - they do NOT reimplement training.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider identity - MUST match the paired ``Policy.provider_name``."""

    @abstractmethod
    def validate(self, spec: TrainSpec) -> list[str]:
        """Pure, side-effect-free preflight.

        Return a list of human-readable problems; an empty list means the spec
        is launchable. Implementations SHOULD check: dataset_root has
        ``meta/info.json``; :attr:`TrainSpec.method` is supported and not a
        contradictory combination (``lora`` + ``expert_only``); any
        backend-required input is present (e.g. a DCP base for Cosmos); and
        rough hardware feasibility against :attr:`hardware_floor`.

        MUST NOT touch the filesystem beyond read-only stat / config reads,
        spawn processes, or allocate GPUs - it powers a ``plan`` advisor that
        runs *before* anything expensive starts.
        """

    def prepare(self, spec: TrainSpec) -> None:
        """Optional one-time setup before :meth:`train`. Default no-op.

        Overridden by backends that need it: Cosmos converts the base
        checkpoint to PyTorch DCP; GR00T registers a modality-config ``.py``.
        LeRobot needs nothing here.
        """
        return None

    @abstractmethod
    def train(self, spec: TrainSpec) -> TrainResult:
        """Build and launch the backend's training, returning a result.

        Responsible for: selecting the launcher (``python`` / ``accelerate
        launch`` / ``torchrun``), mapping :class:`TrainSpec` to native flags,
        wiring resume, and surfacing the checkpoint dir + a status. Long runs
        SHOULD launch detached and return ``status="running"`` with a
        ``job_id`` that :meth:`status` can poll.
        """

    def status(self, job_id: str) -> TrainResult:
        """Return a "RUNNING != learning" verdict for an in-flight job.

        Default implementation returns an ``error`` result indicating the
        backend does not implement status tracking. Backends override to parse
        their training logs for ``latest_step`` / ``latest_loss`` / a
        ``learning`` boolean.
        """
        return TrainResult(
            status="error",
            job_id=job_id,
            message=f"{self.provider_name}: status() not implemented",
        )

    def export(self, spec: TrainSpec, checkpoint_dir: str) -> str:
        """Produce a loadable artifact from a checkpoint.

        Default returns ``checkpoint_dir`` unchanged - correct for HF-native
        backends (LeRobot, GR00T) whose checkpoints are directly loadable by
        ``create_policy(checkpoint_dir)``. Cosmos overrides to convert DCP ->
        safetensors. The returned path MUST be something ``create_policy``
        accepts.
        """
        return checkpoint_dir

    @property
    def hardware_floor(self) -> dict[str, Any]:
        """Advisory minimum hardware, for the ``plan`` advisor.

        Keys: ``min_gpus`` (int), ``min_vram_gb`` (int),
        ``multinode`` (bool). Defaults to a single 24 GB GPU; backends with a
        higher floor (e.g. Cosmos: 8x80 GB) override.
        """
        return {"min_gpus": 1, "min_vram_gb": 24, "multinode": False}
