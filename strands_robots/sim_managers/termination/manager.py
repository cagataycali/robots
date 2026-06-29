"""Termination manager - evaluates termination + timeout terms."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from strands_robots.sim_managers.base import EnvState, Manager, Term, TermSpec, get_term_class


@dataclass
class TerminationResult:
    """Outcome of a :class:`TerminationManager` evaluation.

    Args:
        done: ``True`` if any term (failure or timeout) fired.
        time_out: ``True`` if a timeout term fired (episode truncated, not
            failed). Trainers bootstrap value across a timeout but not a failure.
        terminated: ``True`` if a non-timeout (failure) term fired.
        terms: Per-term boolean breakdown.
    """

    done: bool
    time_out: bool
    terminated: bool
    terms: dict[str, bool]


class TerminationManager(Manager):
    """Evaluate termination terms, separating failures from timeouts.

    Args:
        terms: ``(label, term)`` pairs. A term is treated as a timeout when its
            class sets ``is_time_out = True``.

    Raises:
        ValueError: On duplicate labels.
    """

    category = "termination"

    def compute(self, state: EnvState) -> TerminationResult:
        """Evaluate all terms and classify the episode outcome.

        Args:
            state: The environment state to check.

        Returns:
            A :class:`TerminationResult`.
        """
        breakdown: dict[str, bool] = {}
        time_out = False
        terminated = False
        for label, term in self._terms:
            fired = bool(term(state))
            breakdown[label] = fired
            if not fired:
                continue
            if getattr(term, "is_time_out", False):
                time_out = True
            else:
                terminated = True
        return TerminationResult(
            done=time_out or terminated,
            time_out=time_out,
            terminated=terminated,
            terms=breakdown,
        )

    @classmethod
    def from_specs(cls, specs: Sequence[TermSpec]) -> TerminationManager:
        """Build a :class:`TerminationManager` from term specs.

        Raises:
            ValueError: If a spec references an unknown termination term.
        """
        terms: list[tuple[str, Term]] = []
        for spec in specs:
            term_cls = get_term_class("termination", spec.func)
            terms.append((spec.name, term_cls(**spec.params)))
        return cls(terms)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> TerminationManager:
        """Build from a ``{"terms": [...]}`` config dict."""
        specs = [TermSpec.from_dict(entry) for entry in config.get("terms", [])]
        return cls.from_specs(specs)
