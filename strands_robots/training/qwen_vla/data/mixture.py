"""Weighted multi-source data mixture sampler (Table 1 proportions).

Reproduces the Qwen-VLA pretraining data mixture (arXiv:2605.30280v2 Table 1):
manipulation 74.2% / navigation 7.5% / egocentric 6.0% / simulation 3.7% /
vision-language 8.5%. The sampler draws a source per step according to the
(normalized) weights; users can up-weight their own collected data by adding a
source with a custom weight.

Pure Python + NumPy RNG - no torch, no dataset I/O - so the proportion logic
unit-tests deterministically.
"""

from dataclasses import dataclass

import numpy as np

# Table 1 default mixture proportions (sum ~= 1.0; normalized at runtime).
DEFAULT_MIXTURE: dict[str, float] = {
    "manipulation": 0.742,
    "navigation": 0.075,
    "egocentric": 0.060,
    "simulation": 0.037,
    "vision_language": 0.085,
}


@dataclass
class MixtureSource:
    """One weighted data source in the mixture.

    Attributes:
        name: Source identifier (e.g. ``"manipulation"`` or a user dataset name).
        weight: Relative sampling weight (>= 0; normalized across sources).
        size: Optional number of samples in the source (for length estimates).
    """

    name: str
    weight: float
    size: int = 0

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError(f"source '{self.name}' weight must be >= 0, got {self.weight}")


class MixtureSampler:
    """Draws data-source names according to normalized weights.

    Args:
        sources: List of :class:`MixtureSource`. Weights need not sum to 1 -
            they are normalized. At least one source must have positive weight.
        seed: RNG seed for reproducible draws.

    Raises:
        ValueError: If *sources* is empty or all weights are zero.
    """

    def __init__(self, sources: list[MixtureSource], seed: int = 0):
        if not sources:
            raise ValueError("MixtureSampler requires at least one source")
        total = sum(s.weight for s in sources)
        if total <= 0:
            raise ValueError("at least one source must have a positive weight")

        self.sources = sources
        self._names = [s.name for s in sources]
        self._probs = np.array([s.weight / total for s in sources], dtype=np.float64)
        self._rng = np.random.default_rng(seed)

    @classmethod
    def from_default_mixture(cls, seed: int = 0, **overrides: float) -> "MixtureSampler":
        """Build a sampler from the Table 1 default proportions.

        Args:
            seed: RNG seed.
            **overrides: Per-source weight overrides (e.g.
                ``manipulation=0.8`` to up-weight your own manip data).

        Returns:
            A :class:`MixtureSampler` over the (possibly overridden) defaults.
        """
        weights = dict(DEFAULT_MIXTURE)
        weights.update(overrides)
        sources = [MixtureSource(name=n, weight=w) for n, w in weights.items()]
        return cls(sources, seed=seed)

    @property
    def probabilities(self) -> dict[str, float]:
        """Return the normalized sampling probability per source."""
        return dict(zip(self._names, self._probs.tolist(), strict=True))

    def sample(self) -> str:
        """Draw a single source name according to the weights."""
        idx = self._rng.choice(len(self._names), p=self._probs)
        return self._names[int(idx)]

    def sample_batch(self, n: int) -> list[str]:
        """Draw *n* source names (i.i.d.) according to the weights."""
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        idxs = self._rng.choice(len(self._names), size=n, p=self._probs)
        return [self._names[int(i)] for i in idxs]


__all__ = ["DEFAULT_MIXTURE", "MixtureSource", "MixtureSampler"]
