"""Trainer abstraction - post-tune any policy provider natively.

:class:`Trainer` is the training-side peer of
:class:`~strands_robots.policies.base.Policy`: where ``Policy`` hides how a
model produces actions, ``Trainer`` hides how a model is post-tuned. The
pipelines differ per provider:

* **LeRobot** - build a ``TrainPipelineConfig``, call
  ``lerobot.scripts.lerobot_train.train(cfg)``. HF-native checkpoints.
* **GR00T N1.7** - build a ``FinetuneConfig`` -> ``Config``, call
  ``gr00t.experiment.experiment.run(config)``.
* **Cosmos3** - build the SFT ``Config`` via ``load_experiment_from_toml``,
  call ``cosmos_framework.scripts.train.launch(config, args)``, with a DCP
  checkpoint conversion prepare step and a DCP -> safetensors export step.
* **SageMaker** - submit the same spec as one managed AWS training job.

The first three are *local*: they run in-process and multi-GPU goes through
torch's programmatic ``elastic_launch``. SageMaker is pure *transport*: it
imports no training library and its run outlives the submitting process, which
is what decides each shape's :meth:`Trainer.train` return contract.

All of them converge on one dataset format - LeRobotDataset v3, what
:class:`~strands_robots.dataset_recorder.DatasetRecorder` writes - and one
lifecycle: ``validate -> prepare -> train -> export``. A ``Trainer`` is
selected by the same provider name as its ``Policy``, so one registry identity
owns both classes. See :class:`~strands_robots.training.mock.MockTrainer` for
the no-dependency reference implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainSpec:
    """Provider-agnostic post-tuning specification.

    Concrete trainers read the fields they support and **ignore the rest** -
    the same tolerance rule :meth:`Policy.get_actions` applies to its
    ``**kwargs``. Backends MUST NOT raise on a field they do not use; new
    backend-specific knobs live in :attr:`extra` until >=2 backends share them.
    A backend that reads a field MUST preflight it through the matching
    ``Trainer._*_problems`` gate rather than judge it itself.

    Attributes:
        dataset_root: LeRobotDataset v3 root (contains ``meta/info.json``).
            Optional when :attr:`dataset_repo_id` is set, where it acts as the
            local cache root.
        dataset_repo_id: Hub dataset id (``org/name``) to train from the Hub
            instead of a local root. Supply exactly one data source.
        streaming: Stream frames instead of materializing the dataset. A
            ``bool``. LeRobot-only, and mutually exclusive there with
            :attr:`val_episodes`: a held-out split rebuilds both splits as
            map-style datasets and drops the stream.
        base_model: HF model id or local checkpoint path to post-tune from.
        output_dir: Directory for checkpoints, logs and the final artifact.
        embodiment: Embodiment tag / robot id - which state/action projector
            head the run trains. Required by GR00T. On LeRobot it is read by
            the policies whose config declares ``embodiment_tag`` (GR00T's
            native port); every other LeRobot policy takes its state/action
            shape from the dataset features and has no such field, so a
            backend MUST refuse the request rather than train the default head
            while reporting success.
        steps: Total optimizer steps. A positive ``int``.
        global_batch_size: Batch summed across GPUs before grad accumulation.
            A positive ``int``.
        learning_rate: Optimizer learning rate, or ``None`` for the backend's
            own default. When given, a positive finite number: ``0`` trains the
            full run without updating a weight and ``inf`` checkpoints ``NaN``.
        save_freq: Checkpoint cadence in steps, consumed as the modulus of a
            ``step % save_freq`` test. A non-negative whole number;
            non-positive disables periodic saving so only the final checkpoint
            is written.
        num_gpus: GPUs on this node. A positive ``int``; ``>1`` runs the
            backend under torch's in-process ``elastic_launch``.
        num_nodes: Nodes for multi-node training. Same domain as ``num_gpus``.
        resume: Resume from the latest checkpoint under ``output_dir``. A
            ``bool``.
        seed: Master seed (best-effort). A non-negative ``int`` or ``None``.
        method: Tuning strategy - ``"full"`` | ``"lora"`` | ``"expert_only"``
            | ``"frozen_backbone"``. ``lora`` and ``expert_only`` are mutually
            exclusive (both freeze the VLM).
        lora_r: LoRA adapter rank. A positive ``int``, or ``None`` for peft's
            default. Read only when ``method == "lora"``.
        lora_alpha: Numerator of the ``lora_alpha / lora_r`` scaling. Same
            domain as ``lora_r``.
        lora_target_modules: Target modules, or ``None`` for the policy's
            built-in defaults.
        tune: Component toggles for backends that expose them (GR00T:
            ``{"llm", "visual", "projector", "diffusion"} -> bool``, both
            through Isaac-GR00T's ``--tune_*`` flags and through LeRobot's
            native ``GrootConfig.tune_*`` fields). A key naming no component,
            or a component the policy cannot freeze, MUST be refused: an
            unforwarded toggle trains the config default, which is
            indistinguishable from never having asked.
        val_episodes: Hold out the last N episodes as a validation set, or
            ``None`` to train on every episode. A positive ``int`` below the
            dataset's episode count, which comes from a local
            ``meta/info.json`` - a backend that cannot read one MUST refuse
            rather than emit no split. The count becomes a real-valued split
            fraction whose ceiling lerobot takes. A backend MUST make the
            reserved episodes produce a validation signal, not merely shrink
            the training set.
        augmentation: Backend-specific data augmentation (GR00T
            ``color_jitter_params`` / ``random_rotation_angle``; Cosmos
            dataset filter dict).
        fps: Dataset control rate, when a backend needs it explicitly.
        extra: Raw passthrough; keys become backend-native flags or overrides
            (lerobot ``--key=value``, Cosmos Hydra ``key.path=value``). A value
            may be given as text or as the destination field's own Python type,
            and a backend that assigns into a typed config MUST decode text
            with the same decoder its ``--key=value`` form uses. LeRobot uses
            draccus: ``false``/``no``/``off`` and ``true``/``yes``/``on`` are
            booleans in any case, ``0`` and ``1`` are not, and text that does
            not decode to the field's type is refused.
    """

    # --- universal ---
    dataset_root: str = ""
    base_model: str = ""
    output_dir: str = ""
    dataset_repo_id: str | None = None
    embodiment: str | None = None
    steps: int = 10_000
    global_batch_size: int = 32
    learning_rate: float | None = None
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
    streaming: bool = False
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
    one-time setup) -> :meth:`train` (run + collect verdict) -> :meth:`export`
    (produce a loadable artifact). :meth:`latest_checkpoint` discovers the
    artifact a run produced; :meth:`status` is an optional best-effort verdict
    for a still-running job.

    Concrete trainers come in two shapes and neither reimplements training. A
    **local** trainer imports the backend package and calls its own training
    function in-process (LeRobot ``train(cfg)``, GR00T
    ``experiment.run(config)``, Cosmos ``train.launch(config, args)``), with
    multi-GPU driven by torch's programmatic ``elastic_launch``. A
    **transport** trainer imports no training library: it submits the same
    :class:`TrainSpec` to a managed runner whose image packages a local trainer.
    Only the local shape necessarily finishes inside the :meth:`train` call.

    The ``_*_problems`` methods are the shared field-scoped preflights. Each
    forwards to the matching gate in
    :mod:`strands_robots.training._validate` (imported inside the method to
    keep the ``base -> _validate`` import one-way at runtime) and each is
    obliged only of a backend that *reads* the field - by naming it
    (``spec.seed``) or by forwarding it through a table
    (``getattr(spec, field)``). What obliges the gate is that the value reaches
    the run, not the syntax that fetched it.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider identity - MUST match the paired ``Policy.provider_name``."""

    @abstractmethod
    def validate(self, spec: TrainSpec) -> list[str]:
        """Return the problems that make *spec* unlaunchable; empty when it is.

        A pure preflight: it MUST NOT touch the filesystem beyond read-only
        stat / config reads, spawn processes, or allocate GPUs, because it
        powers a ``plan`` advisor that runs before anything expensive starts.
        Implementations SHOULD check that ``dataset_root`` has
        ``meta/info.json``, that :attr:`TrainSpec.method` is supported and not
        a contradictory combination, that any backend-required input is
        present, and rough feasibility against :meth:`hardware_floor`.
        """

    def _security_problems(self, spec: TrainSpec) -> list[str]:
        """Preflight every agent-supplied value for input safety.

        Path traversal and protected directories, a leading ``-`` a backend's
        argv-parity helper would read as a flag, and an ``extra`` key that
        would set an arbitrary config attribute or Hydra override. Concrete
        :meth:`validate` implementations MUST call this first, before any
        config is built.
        """
        from strands_robots.training._validate import validate_train_inputs

        return validate_train_inputs(spec)

    def _run_size_problems(self, spec: TrainSpec) -> list[str]:
        """Preflight ``steps`` and ``global_batch_size``, each a positive ``int``."""
        from strands_robots.training._validate import run_size_problems

        return run_size_problems(spec, context=self.provider_name)

    def _rl_run_size_problems(self, spec: TrainSpec) -> list[str]:
        """Preflight ``total_timesteps`` and ``rollout_steps``, each a positive ``int``.

        ``num_envs`` is graded by the caller that reads it: which counts are
        usable differs per backend, while being a count at all does not.
        """
        from strands_robots.training._validate import rl_run_size_problems

        return rl_run_size_problems(spec, context=self.provider_name)

    def _rl_replay_problems(self, spec: TrainSpec) -> list[str]:
        """Preflight ``buffer_size``, ``batch_size`` and ``gradient_steps``.

        Each a positive ``int``. A zero ``gradient_steps`` takes zero gradient
        updates for the whole run and still reports success with a checkpoint.
        """
        from strands_robots.training._validate import rl_replay_problems

        return rl_replay_problems(spec, context=self.provider_name)

    def _learning_rate_problems(self, spec: TrainSpec) -> list[str]:
        """Preflight ``learning_rate`` when supplied: a positive finite number."""
        from strands_robots.training._validate import learning_rate_problems

        return learning_rate_problems(spec, context=self.provider_name)

    def _launch_topology_problems(self, spec: TrainSpec) -> list[str]:
        """Preflight ``num_gpus`` and ``num_nodes``, each a positive ``int``."""
        from strands_robots.training._validate import launch_topology_problems

        return launch_topology_problems(spec, context=self.provider_name)

    def _seed_problems(self, spec: TrainSpec) -> list[str]:
        """Preflight ``seed`` when supplied: a non-negative ``int``."""
        from strands_robots.training._validate import seed_problems

        return seed_problems(spec, context=self.provider_name)

    def _checkpoint_cadence_problems(self, spec: TrainSpec) -> list[str]:
        """Preflight ``save_freq``: a non-negative whole number of steps."""
        from strands_robots.training._validate import checkpoint_cadence_problems

        return checkpoint_cadence_problems(spec, context=self.provider_name)

    def _rl_checkpoint_interval_problems(self, spec: TrainSpec) -> list[str]:
        """Preflight ``log_interval``: a non-negative whole number of iterations."""
        from strands_robots.training._validate import rl_checkpoint_interval_problems

        return rl_checkpoint_interval_problems(spec, context=self.provider_name)

    def _validation_episodes_problems(self, spec: TrainSpec) -> list[str]:
        """Preflight ``val_episodes`` when supplied: a positive ``int``.

        The count becomes a real-valued split fraction whose ceiling lerobot
        takes, so a backend MUST NOT compare it itself.
        """
        from strands_robots.training._validate import validation_episodes_problems

        return validation_episodes_problems(spec, context=self.provider_name)

    def _resume_problems(self, spec: TrainSpec) -> list[str]:
        """Preflight ``resume``: a ``bool``."""
        from strands_robots.training._validate import resume_problems

        return resume_problems(spec, context=self.provider_name)

    def _streaming_problems(self, spec: TrainSpec) -> list[str]:
        """Preflight ``streaming``: a ``bool``.

        A backend whose ``validate`` branches on the field MUST consult this
        first and read the field only when it reports nothing, so a misread
        posture is refused by the flag's own name.
        """
        from strands_robots.training._validate import streaming_problems

        return streaming_problems(spec, context=self.provider_name)

    def _lora_hyperparameter_problems(self, spec: TrainSpec) -> list[str]:
        """Preflight ``lora_r`` and ``lora_alpha``, each a positive ``int`` or ``None``.

        peft judges only the rank, and only once the base model is loaded.
        """
        from strands_robots.training._validate import lora_hyperparameter_problems

        return lora_hyperparameter_problems(spec, context=self.provider_name)

    def _discount_factor_problems(self, spec: TrainSpec) -> list[str]:
        """Preflight ``gamma``: a real in ``[0, 1]``. Every RL backend reads it."""
        from strands_robots.training._validate import discount_factor_problems

        return discount_factor_problems(spec, context=self.provider_name)

    def _gae_lambda_problems(self, spec: TrainSpec) -> list[str]:
        """Preflight ``lam``: a real in ``[0, 1]``, for a backend that estimates a trace.

        The trace decays by the product ``gamma * lam``, so bounding one factor
        does not bound the trace.
        """
        from strands_robots.training._validate import gae_lambda_problems

        return gae_lambda_problems(spec, context=self.provider_name)

    def _optimization_epochs_problems(self, spec: TrainSpec) -> list[str]:
        """Preflight ``num_learning_epochs``: a positive ``int``.

        It is the ``range()`` bound enclosing every ``optimizer.step()``, so a
        non-positive value takes no gradient step while the run reports success
        and writes an untrained checkpoint.
        """
        from strands_robots.training._validate import optimization_epochs_problems

        return optimization_epochs_problems(spec, context=self.provider_name)

    def _temperature_learning_rate_problems(self, spec: TrainSpec) -> list[str]:
        """Preflight ``alpha_lr``: a positive finite number, when a temperature is tuned."""
        from strands_robots.training._validate import temperature_learning_rate_problems

        return temperature_learning_rate_problems(spec, context=self.provider_name)

    def _initial_temperature_problems(self, spec: TrainSpec) -> list[str]:
        """Preflight ``init_alpha``: a positive finite number.

        The temperature is stored as its logarithm, so a non-positive value has
        none. Not scoped to ``autotune_alpha`` - the coercion is unconditional.
        """
        from strands_robots.training._validate import initial_temperature_problems

        return initial_temperature_problems(spec, context=self.provider_name)

    def _target_entropy_problems(self, spec: TrainSpec) -> list[str]:
        """Preflight ``target_entropy``: a finite real of either sign, or ``None``.

        The domain is signed because the field defaults to ``-num_actions``, so
        no endpoint is decidable and ``True`` is a sign flip rather than merely
        a flag read as a number. ``None`` requests that heuristic default. A
        backend that builds FastSAC's temperature block MUST call this
        alongside :meth:`_initial_temperature_problems` and
        :meth:`_temperature_learning_rate_problems`: those guard the starting
        value and the rate that moves it, this one the constant it moves
        toward.
        """
        from strands_robots.training._validate import target_entropy_problems

        return target_entropy_problems(spec, context=self.provider_name)

    def _polyak_coefficient_problems(self, spec: TrainSpec) -> list[str]:
        """Preflight ``tau``: a real in ``(0, 1]``.

        A zero coefficient never moves the target networks, so the TD target is
        the untrained initialisation for the whole run.
        """
        from strands_robots.training._validate import polyak_coefficient_problems

        return polyak_coefficient_problems(spec, context=self.provider_name)

    def _spec_device_problems(self, spec: TrainSpec) -> list[str]:
        """Preflight ``device``: one torch can place tensors on, when stated."""
        from strands_robots.training._validate import torch_device_problems

        return torch_device_problems(spec, context=self.provider_name)

    def _gradient_clip_problems(self, spec: TrainSpec) -> list[str]:
        """Preflight ``max_grad_norm``: a positive real, ``inf`` meaning do not clip."""
        from strands_robots.training._validate import gradient_clip_problems

        return gradient_clip_problems(spec, context=self.provider_name)

    def _loss_weight_problems(self, spec: TrainSpec) -> list[str]:
        """Preflight ``value_loss_coef`` and ``entropy_coef``, each a finite real.

        Zero and negative are real configurations for both, so only the domain
        is bounded, not the floor.
        """
        from strands_robots.training._validate import loss_weight_problems

        return loss_weight_problems(spec, context=self.provider_name)

    def _clip_range_problems(self, spec: TrainSpec) -> list[str]:
        """Preflight ``clip_param``: a positive real, ``inf`` meaning do not clip.

        It is the half-width of the trust region and also clips the value loss.
        """
        from strands_robots.training._validate import clip_range_problems

        return clip_range_problems(spec, context=self.provider_name)

    def _policy_delay_problems(self, spec: TrainSpec) -> list[str]:
        """Preflight ``policy_delay``: a positive ``int``.

        It is the modulus of an ``update_count % policy_delay`` test, so a
        value that never satisfies it trains the critics while the deployable
        actor never takes a gradient step.
        """
        from strands_robots.training._validate import policy_delay_problems

        return policy_delay_problems(spec, context=self.provider_name)

    def _td3_noise_problems(self, spec: TrainSpec) -> list[str]:
        """Preflight the three TD3 noise scalars, each a positive finite number.

        ``exploration_noise_std``, ``target_noise_std`` and
        ``target_noise_clip``. Gaussian noise is symmetric, so a negative scale
        is the identical distribution and zero removes the mechanism.
        """
        from strands_robots.training._validate import td3_noise_problems

        return td3_noise_problems(spec, context=self.provider_name)

    def _network_width_problems(self, spec: TrainSpec) -> list[str]:
        """Preflight ``hidden_dims``: a sequence of positive ``int`` layer widths."""
        from strands_robots.training._validate import network_width_problems

        return network_width_problems(spec, context=self.provider_name)

    def prepare(self, spec: TrainSpec) -> None:
        """Optional one-time setup before :meth:`train`. Default no-op.

        Cosmos converts the base checkpoint to PyTorch DCP; GR00T registers a
        modality-config ``.py``; LeRobot needs nothing here.
        """
        return None

    @abstractmethod
    def train(self, spec: TrainSpec) -> TrainResult:
        """Run the backend's training and return the result of that run.

        Builds the backend's typed config from *spec*, wires resume, selects
        single- vs multi-GPU (``elastic_launch`` for ``num_gpus > 1``), invokes
        the backend's own training function and surfaces the checkpoint dir plus
        the metrics verdict. Every implementation MUST call :meth:`validate`
        first and fail closed.

        A *local* trainer blocks until the run finishes and returns a terminal
        :class:`TrainResult` with ``metrics`` populated. A *transport* trainer
        MAY return ``running`` with a ``job_id`` that :meth:`status` polls and
        no ``checkpoint_dir`` yet, so a caller must branch on all three
        ``status`` values rather than read "not ``error``" as finished.
        """

    def status(self, job_id: str) -> TrainResult:
        """Return a "RUNNING != learning" verdict for a job still in flight.

        Two kinds of job reach here: one launched out of band that a caller
        polls by id, and one a *transport* :meth:`train` handed back as
        ``running`` because it outlives the submitting process. A local trainer
        produces neither, so
        most backends inherit this default, which returns an informative
        ``error``. Backends that override read the runner's own job API
        (``sagemaker`` -> ``DescribeTrainingJob``) or parse their training logs.
        """
        return TrainResult(
            status="error",
            job_id=job_id,
            message=(
                f"{self.provider_name}: status() polling is not supported - "
                "train() runs synchronously and already returns the metrics verdict."
            ),
        )

    def export(self, spec: TrainSpec, checkpoint_dir: str) -> str:
        """Produce a loadable artifact from *checkpoint_dir* and return its path.

        The default returns ``checkpoint_dir`` unchanged, which is correct for
        HF-native backends whose checkpoints ``create_policy`` loads directly;
        Cosmos overrides to convert DCP -> safetensors. The returned path MUST
        be something ``create_policy`` accepts.
        """
        return checkpoint_dir

    def latest_checkpoint(self, output_dir: str) -> str | None:
        """Return the newest loadable checkpoint directory under ``output_dir``.

        ``None`` when no checkpoint exists yet or the backend writes no
        discoverable tree, which is the default. Read-only (stat only).
        """
        return None

    @property
    def hardware_floor(self) -> dict[str, Any]:
        """Return the advisory minimum hardware for the ``plan`` advisor.

        Keys ``min_gpus`` (int), ``min_vram_gb`` (int), ``multinode`` (bool).
        Defaults to a single 24 GB GPU; a backend with a higher floor (Cosmos:
        8x80 GB) overrides.
        """
        return {"min_gpus": 1, "min_vram_gb": 24, "multinode": False}
