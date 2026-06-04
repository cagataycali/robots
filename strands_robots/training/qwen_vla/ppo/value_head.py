"""Value head spec for Qwen-VLA PPO (section 4.2).

A lightweight value head attached to the VLM hidden states with a
**stop-gradient** on the backbone (the paper runs the value head at ~20x the
actor LR). This module holds the torch-free *spec* (dimensions, LR multiplier,
stop-grad flag) so it can be validated and serialized without torch; the actual
``nn.Module`` is built by the training model from this spec.
"""

from dataclasses import dataclass


@dataclass
class ValueHeadSpec:
    """Configuration for the VLM-attached value head.

    Attributes:
        hidden_dim: Width of the VLM hidden states fed to the head.
        mlp_dims: Hidden layer widths of the value MLP (output is scalar).
        stop_gradient: Stop gradients into the VLM backbone (paper default
            True - the value head must not corrupt the actor's representations).
        lr_multiplier: Value-head LR as a multiple of the actor LR (paper ~20x).
    """

    hidden_dim: int
    mlp_dims: tuple[int, ...] = (256, 256)
    stop_gradient: bool = True
    lr_multiplier: float = 20.0

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if any(d <= 0 for d in self.mlp_dims):
            raise ValueError(f"all mlp_dims must be positive, got {self.mlp_dims}")
        if self.lr_multiplier <= 0:
            raise ValueError(f"lr_multiplier must be positive, got {self.lr_multiplier}")

    def value_lr(self, actor_lr: float) -> float:
        """Return the value-head LR for a given actor LR."""
        return actor_lr * self.lr_multiplier


__all__ = ["ValueHeadSpec"]
