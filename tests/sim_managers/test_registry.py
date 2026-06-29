"""Term registry: lookup, listing, and duplicate / unknown handling."""

from __future__ import annotations

import pytest

from strands_robots.sim_managers import get_term_class, list_terms, register_term
from strands_robots.sim_managers.base import Term


def test_locomotion_terms_registered():
    terms = list_terms()
    assert "track_lin_vel_xy_exp" in terms["reward"]
    assert "base_lin_vel" in terms["observation"]
    assert "time_out" in terms["termination"]
    assert "uniform_velocity" in terms["command"]


def test_unknown_term_lists_available():
    with pytest.raises(ValueError, match="available"):
        get_term_class("reward", "does_not_exist")


def test_unknown_category_raises():
    with pytest.raises(ValueError, match="unknown term category"):
        get_term_class("nope", "x")


def test_duplicate_registration_raises():
    with pytest.raises(ValueError, match="already registered"):

        @register_term("reward", "track_lin_vel_xy_exp")
        class _Dup(Term):
            def __call__(self, state):  # pragma: no cover - registration fails first
                return 0.0
