"""Observation manager - composes observation terms into one policy vector."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from strands_robots.sim_managers.base import EnvState, FloatArray, Manager, Term, TermSpec, get_term_class


class ObservationManager(Manager):
    """Concatenate observation terms into a single 1-D observation vector.

    Each term is scaled then optionally clipped (per :class:`TermSpec`) before
    concatenation, in declaration order. The resulting vector layout is stable
    and introspectable via :attr:`term_slices`.

    Args:
        terms: ``(label, term, scale, clip)`` tuples in observation order.

    Raises:
        ValueError: On duplicate labels.
    """

    category = "observation"

    def __init__(self, terms: list[tuple[str, Term, float, tuple[float, float] | None]]) -> None:
        super().__init__([(label, term) for label, term, _, _ in terms])
        self._scales: dict[str, float] = {label: scale for label, _, scale, _ in terms}
        self._clips: dict[str, tuple[float, float] | None] = {label: clip for label, _, _, clip in terms}
        self._slices: dict[str, slice] = {}

    @property
    def term_slices(self) -> dict[str, slice]:
        """Map term label -> slice into the observation vector (after first compute)."""
        return dict(self._slices)

    def compute(self, state: EnvState) -> FloatArray:
        """Return the concatenated, scaled, clipped observation vector.

        Args:
            state: The environment state to observe.

        Returns:
            A 1-D ``float64`` observation vector. Empty (shape ``(0,)``) when the
            manager has no terms.
        """
        chunks: list[FloatArray] = []
        offset = 0
        self._slices = {}
        for label, term in self._terms:
            value = np.atleast_1d(np.asarray(term(state), dtype=np.float64))
            value = value * self._scales[label]
            clip = self._clips[label]
            if clip is not None:
                value = np.clip(value, clip[0], clip[1])
            self._slices[label] = slice(offset, offset + value.shape[0])
            offset += value.shape[0]
            chunks.append(value)
        if not chunks:
            return np.zeros(0)
        return np.concatenate(chunks)

    @classmethod
    def from_specs(cls, specs: Sequence[TermSpec]) -> ObservationManager:
        """Build an :class:`ObservationManager` from term specs.

        Args:
            specs: Ordered observation term specs.

        Returns:
            The constructed manager.

        Raises:
            ValueError: If a spec references an unknown observation term.
        """
        terms: list[tuple[str, Term, float, tuple[float, float] | None]] = []
        for spec in specs:
            term_cls = get_term_class("observation", spec.func)
            terms.append((spec.name, term_cls(**spec.params), spec.scale, spec.clip))
        return cls(terms)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ObservationManager:
        """Build from a ``{"terms": [...]}`` config dict.

        Args:
            config: Manager config with a ``terms`` list.

        Returns:
            The constructed manager.
        """
        specs = [TermSpec.from_dict(entry) for entry in config.get("terms", [])]
        return cls.from_specs(specs)
