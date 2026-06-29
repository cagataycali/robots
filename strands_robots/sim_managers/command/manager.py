"""Command manager - generates and refreshes command vectors on the EnvState."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from strands_robots.sim_managers.base import EnvState, FloatArray, Manager, Term, TermSpec, get_term_class


class CommandManager(Manager):
    """Generate named command vectors and write them onto the :class:`EnvState`.

    Calling :meth:`compute` advances each command term's resample timer by
    ``state.dt`` (when the term supports it) and writes the resulting vector into
    ``state.commands[label]``, so downstream reward / observation terms can read
    it via ``state.command(label)``.

    Args:
        terms: ``(label, term)`` pairs.

    Raises:
        ValueError: On duplicate labels.
    """

    category = "command"

    def compute(self, state: EnvState) -> dict[str, FloatArray]:
        """Refresh and publish all commands onto ``state.commands``.

        Args:
            state: The environment state to populate.

        Returns:
            The mapping of command label -> vector that was written.
        """
        out: dict[str, FloatArray] = {}
        for label, term in self._terms:
            update = getattr(term, "update", None)
            if callable(update):
                update(state.dt)
            value = term(state)
            state.commands[label] = value
            out[label] = value
        return out

    @classmethod
    def from_specs(cls, specs: Sequence[TermSpec]) -> CommandManager:
        """Build a :class:`CommandManager` from term specs.

        Raises:
            ValueError: If a spec references an unknown command term.
        """
        terms: list[tuple[str, Term]] = []
        for spec in specs:
            term_cls = get_term_class("command", spec.func)
            terms.append((spec.name, term_cls(**spec.params)))
        return cls(terms)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> CommandManager:
        """Build from a ``{"terms": [...]}`` config dict."""
        specs = [TermSpec.from_dict(entry) for entry in config.get("terms", [])]
        return cls.from_specs(specs)
