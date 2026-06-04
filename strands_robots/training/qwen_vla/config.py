"""Configuration dataclasses for the Qwen-VLA 4-stage training recipe.

Paper defaults (arXiv:2605.30280v2) are encoded as documented constants and
cited inline. Pure dataclasses - no torch - so config can be validated and
serialized without the training extras installed.
"""

from dataclasses import dataclass
from enum import StrEnum


class TimestepDist(StrEnum):
    """Flow-matching timestep sampling distribution (section 5.2.1).

    The paper finds **Sigmoid-Normal** best for T2A (stage 1) and **Beta** for
    CPT (stage 2); both beat uniform on the action-prediction ablations.
    """

    UNIFORM = "uniform"
    BETA = "beta"
    SIGMOID_NORMAL = "sigmoid_normal"


@dataclass
class _BaseStageConfig:
    """Fields shared by every training stage."""

    # Action horizon H. Manipulation = 16, navigation waypoints = 8 (section 2.1).
    chunk_size: int = 16
    # Fixed unified channel width K (widest embodiment; zero-padded, section 2.4).
    action_dim: int = 32
    batch_size: int = 64
    learning_rate: float = 1e-4
    # VL co-training loss weight vs action loss (section 3 / eq. 3): 0.1 / 1.0.
    vl_loss_weight: float = 0.1
    action_loss_weight: float = 1.0
    seed: int = 42
    output_dir: str = "checkpoints/qwen_vla"

    def validate(self) -> None:
        """Raise on invalid configuration (no silent defaults, AGENTS.md)."""
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {self.action_dim}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")


@dataclass
class T2AConfig(_BaseStageConfig):
    """Stage 1: Text-to-Action (T2A) DiT pretraining (section 3.2.3 / 5.2.1).

    VLM frozen, no images, full-sequence prediction (+4.9pp vs chunk per the
    ablation), Sigmoid-Normal timestep dist, ~2k steps sweet spot, ~20%
    synthetic + 80% real mix.
    """

    freeze_vlm: bool = True
    use_images: bool = False
    full_sequence_prediction: bool = True
    timestep_dist: TimestepDist = TimestepDist.SIGMOID_NORMAL
    max_steps: int = 2000
    synthetic_fraction: float = 0.2

    def validate(self) -> None:
        super().validate()
        if not 0.0 <= self.synthetic_fraction <= 1.0:
            raise ValueError(f"synthetic_fraction must be in [0,1], got {self.synthetic_fraction}")
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")


@dataclass
class CPTConfig(_BaseStageConfig):
    """Stage 2: Continued Pretraining (CPT) - joint VLM+DiT (section 5.2.1).

    Unfreezes both modules, trains on the heterogeneous mixture, Beta timestep
    dist, VL co-training to prevent catastrophic forgetting. Produces
    Qwen-VLA-Base.
    """

    freeze_vlm: bool = False
    use_images: bool = True
    timestep_dist: TimestepDist = TimestepDist.BETA
    max_steps: int = 20000
    warmup_checkpoint: str | None = None  # T2A stage-1 decoder warm-start

    def validate(self) -> None:
        super().validate()
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")


@dataclass
class SFTConfig(_BaseStageConfig):
    """Stage 3: Supervised Fine-Tuning (SFT) on curated demos (section 4).

    Two tracks: multi-task (VQA + grounding + manip + nav) and real-robot
    teleop from DatasetRecorder output. H=16 manip / 8 nav waypoints.
    """

    use_images: bool = True
    timestep_dist: TimestepDist = TimestepDist.BETA
    max_steps: int = 10000
    base_checkpoint: str | None = None  # Qwen-VLA-Base from CPT
    multi_task: bool = True
    nav_chunk_size: int = 8

    def validate(self) -> None:
        super().validate()
        if self.nav_chunk_size <= 0:
            raise ValueError(f"nav_chunk_size must be positive, got {self.nav_chunk_size}")


@dataclass
class RLConfig(_BaseStageConfig):
    """Stage 4: PPO + GAE on sim success reward (section 4.2). Produces Instruct.

    Flow-matching logpi via probability-flow ODE->SDE; value head on VLM hidden
    states with stop-gradient (value LR ~ 20x actor LR); action-chunk-level
    credit assignment (one scalar reward + advantage per H chunk).
    """

    timestep_dist: TimestepDist = TimestepDist.BETA
    sft_checkpoint: str | None = None  # Qwen-VLA from SFT
    num_envs: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    ppo_epochs: int = 4
    value_lr_multiplier: float = 20.0
    rollout_steps: int = 256

    def validate(self) -> None:
        super().validate()
        if self.num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {self.num_envs}")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError(f"gamma must be in (0,1], got {self.gamma}")
        if not 0.0 < self.gae_lambda <= 1.0:
            raise ValueError(f"gae_lambda must be in (0,1], got {self.gae_lambda}")
        if self.clip_epsilon <= 0:
            raise ValueError(f"clip_epsilon must be positive, got {self.clip_epsilon}")


__all__ = ["TimestepDist", "T2AConfig", "CPTConfig", "SFTConfig", "RLConfig"]
