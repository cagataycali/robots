"""``BaseRLAlgo`` - the from-scratch reinforcement-learning peer of ``Trainer``.

The supervised :class:`~strands_robots.training.base.Trainer` post-tunes a model
*from a dataset*. RL is the missing other half: train a policy *from a reward
function* by interacting with a simulation. ``BaseRLAlgo`` is a ``Trainer``
subclass so it is selected through the SAME
:func:`~strands_robots.training.factory.create_trainer` factory
(``create_trainer("ppo")``), but it adds the RL lifecycle on top of the
``validate -> train -> export`` contract:

    setup(spec) -> [ collect_rollout() -> update() ]* -> save_checkpoint()

``train()`` is implemented here as the standard on-policy loop over those hooks,
so a concrete on-policy algorithm (PPO) only implements the four hooks. An
off-policy algorithm (SAC) overrides ``train()`` with its own
replay-buffer loop while keeping the same hooks and checkpoint contract.

Adapted in spirit from the Amazon FAR Holosoma ``BaseAlgo`` (BSD-3-Clause,
https://github.com/amazon-far/holosoma), re-homed onto the strands-robots
``SimEngine`` env interface instead of IsaacGym.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from strands_robots.training.base import Trainer, TrainResult, TrainSpec

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

    from strands_robots.training.rl.env import SimEnv


@dataclass
class RLTrainSpec(TrainSpec):
    """RL extension of :class:`TrainSpec` - reward-driven, not dataset-driven.

    RL ignores the dataset fields of ``TrainSpec`` (``dataset_root`` etc.) and
    drives training from an environment + reward instead. The supervised fields
    remain so the one factory / lifecycle is shared; RL backends read only the
    fields below plus the universal ``output_dir`` / ``learning_rate`` /
    ``seed`` / ``num_gpus``.

    Attributes:
        env_factory: Zero-arg callable returning a freshly-built
            :class:`~strands_robots.training.rl.env.SimEnv`. A factory (not an
            instance) so the trainer owns the env lifecycle and a future
            vectorized backend can build N of them.
        total_timesteps: Total environment steps to train for. The number of
            policy-update iterations is ``total_timesteps // (rollout_steps *
            num_envs)``.
        rollout_steps: Environment steps collected per iteration before each
            policy update (the on-policy batch horizon, holosoma ``num_steps``).
        num_envs: Parallel environments. ``1`` for the MuJoCo single-env
            backend; vectorized backends raise it.
        actor_obs_keys / critic_obs_keys: Documentation of the observation
            contract (the env enforces it); kept on the spec so a plan/advisor
            can echo it without constructing the env.
        gamma: Discount factor.
        lam: GAE-lambda.
        clip_param: PPO clip range (also clips the value loss).
        num_learning_epochs: Optimization epochs over each rollout batch.
        num_mini_batches: Minibatches the rollout batch is split into per epoch.
        entropy_coef: Entropy-bonus weight (exploration).
        value_loss_coef: Value-loss weight.
        max_grad_norm: Gradient-norm clip.
        hidden_dims: MLP hidden layer sizes for actor and critic.
        init_noise_std: Initial action-distribution standard deviation.
        normalize_obs: Wrap observations in ``EmpiricalNormalization``.
        normalize_advantage: Standardize advantages per batch.
        device: Torch device (``"cpu"`` / ``"cuda"``); ``None`` auto-selects.
        log_interval: Iterations between progress logs.
        buffer_size: Off-policy replay-buffer capacity (SAC).
        batch_size: Transitions sampled per gradient step (SAC).
        learning_starts: Env steps of random warmup collected into the
            buffer before the first gradient update (SAC).
        gradient_steps: SAC gradient updates run per training iteration.
        tau: Polyak averaging coefficient for the target critics (SAC).
        autotune_alpha: Automatically tune the entropy temperature against
            ``target_entropy`` (SAC).
        init_alpha: Initial entropy temperature (SAC).
        alpha_lr: Learning rate for the temperature optimizer (SAC).
        target_entropy: Target policy entropy; ``None`` uses ``-num_actions``
            (the SAC heuristic).
    """

    env_factory: Callable[[], SimEnv] | None = None
    total_timesteps: int = 100_000
    rollout_steps: int = 24
    num_envs: int = 1
    actor_obs_keys: list[str] = field(default_factory=list)
    critic_obs_keys: list[str] = field(default_factory=list)
    gamma: float = 0.99
    lam: float = 0.95
    clip_param: float = 0.2
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    entropy_coef: float = 0.0
    value_loss_coef: float = 1.0
    max_grad_norm: float = 1.0
    hidden_dims: tuple[int, ...] = (128, 128)
    init_noise_std: float = 1.0
    normalize_obs: bool = True
    normalize_advantage: bool = True
    device: str | None = None
    log_interval: int = 10
    # --- off-policy (SAC) fields; ignored by on-policy backends (PPO) ---
    buffer_size: int = 100_000
    batch_size: int = 256
    learning_starts: int = 1_000
    gradient_steps: int = 1
    tau: float = 0.005
    autotune_alpha: bool = True
    init_alpha: float = 1.0
    alpha_lr: float = 3e-4
    target_entropy: float | None = None


class BaseRLAlgo(Trainer):
    """Abstract from-scratch RL trainer (peer of supervised ``Trainer``).

    Concrete on-policy algorithms implement :meth:`setup`, :meth:`collect_rollout`,
    :meth:`update`, and :meth:`save_checkpoint`; the default :meth:`train` runs the
    on-policy loop over them. ``steps_per_iter`` (set in :meth:`setup`) is the env
    steps consumed per iteration, used to translate ``total_timesteps`` into an
    iteration count.
    """

    steps_per_iter: int = 1

    @abstractmethod
    def setup(self, spec: RLTrainSpec) -> None:
        """Build the env, networks, optimizer, and rollout storage from ``spec``.

        MUST set :attr:`steps_per_iter` to ``rollout_steps * num_envs``.
        """

    @abstractmethod
    def collect_rollout(self) -> dict[str, float]:
        """Collect one on-policy batch; return rollout metrics (e.g. mean reward)."""

    @abstractmethod
    def update(self) -> dict[str, float]:
        """Run the policy/value update on the collected batch; return loss metrics."""

    @abstractmethod
    def save_checkpoint(self, output_dir: str, iteration: int | None = None) -> str:
        """Persist a loadable checkpoint; return its directory."""

    def train(self, spec: TrainSpec) -> TrainResult:
        """Default on-policy training loop: setup -> (collect, update)* -> save.

        Off-policy algorithms override this. ``spec`` MUST be an
        :class:`RLTrainSpec`; :meth:`validate` is called first and fails closed.
        """
        if not isinstance(spec, RLTrainSpec):
            return TrainResult(
                status="error",
                job_id="",
                message=f"{self.provider_name} requires an RLTrainSpec, got {type(spec).__name__}",
            )
        problems = self.validate(spec)
        if problems:
            return TrainResult(status="error", job_id="", message="validation failed: " + "; ".join(problems))

        self.setup(spec)
        steps_per_iter = max(1, self.steps_per_iter)
        num_iters = max(1, spec.total_timesteps // steps_per_iter)

        job_id = f"{self.provider_name}-{id(self):x}"
        last_metrics: dict[str, Any] = {}
        ckpt_dir: str | None = None
        for it in range(num_iters):
            rollout_metrics = self.collect_rollout()
            loss_metrics = self.update()
            last_metrics = {**rollout_metrics, **loss_metrics, "iteration": it + 1}
            if spec.log_interval and (it % spec.log_interval == 0 or it == num_iters - 1):
                ckpt_dir = self.save_checkpoint(spec.output_dir, iteration=it + 1)
        if ckpt_dir is None:
            ckpt_dir = self.save_checkpoint(spec.output_dir, iteration=num_iters)

        last_metrics.setdefault("latest_step", num_iters * steps_per_iter)
        return TrainResult(
            status="success",
            job_id=job_id,
            checkpoint_dir=ckpt_dir,
            exported_model=self.export(spec, ckpt_dir),
            metrics=last_metrics,
            message=f"{self.provider_name}: {num_iters} iterations x {steps_per_iter} steps complete",
        )
