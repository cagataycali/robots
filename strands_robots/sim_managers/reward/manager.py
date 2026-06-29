"""Reward manager - composes weighted reward terms into a scalar reward."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from strands_robots.sim_managers.base import EnvState, Manager, Term, TermSpec, get_term_class


class RewardManager(Manager):
    """Sum weighted reward terms into a scalar reward per control step.

    Each term contributes ``weight * term(state) * dt`` (Isaac Lab convention:
    rewards are time-integrated so a recipe is invariant to control frequency).
    The most recent per-term contribution is kept in :attr:`term_values` for
    logging / curriculum.

    Args:
        terms: ``(label, term, weight)`` tuples.
        scale_by_dt: When ``True`` (default), multiply each contribution by
            ``state.dt`` so total reward is consistent across control rates.

    Raises:
        ValueError: On duplicate labels.
    """

    category = "reward"

    def __init__(self, terms: list[tuple[str, Term, float]], *, scale_by_dt: bool = True) -> None:
        super().__init__([(label, term) for label, term, _ in terms])
        self._weights: dict[str, float] = {label: weight for label, _, weight in terms}
        self._scale_by_dt = scale_by_dt
        self._term_values: dict[str, float] = {}

    @property
    def term_values(self) -> dict[str, float]:
        """Most recent weighted per-term contributions (after :meth:`compute`)."""
        return dict(self._term_values)

    @property
    def weights(self) -> dict[str, float]:
        """Configured per-term weights."""
        return dict(self._weights)

    def compute(self, state: EnvState) -> float:
        """Return the total weighted reward and refresh :attr:`term_values`.

        Args:
            state: The environment state to score.

        Returns:
            The scalar reward for this step.
        """
        dt = state.dt if self._scale_by_dt else 1.0
        self._term_values = {}
        for label, term in self._terms:
            self._term_values[label] = self._weights[label] * float(term(state)) * dt
        # Return the sum of the recorded breakdown so the contract
        # ``compute(state) == sum(term_values.values())`` holds exactly.
        return float(sum(self._term_values.values()))

    @classmethod
    def from_specs(cls, specs: Sequence[TermSpec], *, scale_by_dt: bool = True) -> RewardManager:
        """Build a :class:`RewardManager` from term specs.

        Args:
            specs: Ordered reward term specs (``weight`` is honoured).
            scale_by_dt: See the constructor.

        Returns:
            The constructed manager.

        Raises:
            ValueError: If a spec references an unknown reward term.
        """
        terms: list[tuple[str, Term, float]] = []
        for spec in specs:
            term_cls = get_term_class("reward", spec.func)
            terms.append((spec.name, term_cls(**spec.params), spec.weight))
        return cls(terms, scale_by_dt=scale_by_dt)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> RewardManager:
        """Build from a ``{"terms": [...], "scale_by_dt": bool}`` config dict."""
        specs = [TermSpec.from_dict(entry) for entry in config.get("terms", [])]
        return cls.from_specs(specs, scale_by_dt=bool(config.get("scale_by_dt", True)))
