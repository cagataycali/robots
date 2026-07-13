"""Newton ``create_world(terrain=...)`` rejects an unsupported terrain kind.

Rough-ground heightfields are a MuJoCo-backend capability; the Newton backend
has no heightfield ground yet. Rather than silently ignoring a ``terrain=``
request (which would spawn a locomotion robot on a flat plane while the caller
believes it is on terrain), Newton rejects a non-None ``terrain`` with an
actionable error pointing at the MuJoCo backend. That rejection is a pure
Python contract that returns before any GPU/model build, so it is exercised
here via ``__new__`` (no Warp / GPU required, runs in CI).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

_engine_cls: type[NewtonSimEngine] | None
try:  # NewtonSimEngine imports without Warp (Warp is lazily loaded at build time)
    from strands_robots.simulation.newton.simulation import NewtonSimEngine as _engine_cls
except Exception:  # pragma: no cover - newton package genuinely absent
    _engine_cls = None

pytestmark = pytest.mark.skipif(_engine_cls is None, reason="newton package not importable")


def test_newton_rejects_terrain_before_any_build() -> None:
    # __new__ bypasses __init__ (no solver/GPU); the reject returns before the
    # lock/_rebuild, so no engine state is needed.
    assert _engine_cls is not None
    eng = _engine_cls.__new__(_engine_cls)
    r = eng.create_world(terrain="rough")
    assert r["status"] == "error"
    text = r["content"][0]["text"]
    assert "Newton" in text and "MuJoCo" in text and "rough" in text


def test_newton_terrain_none_is_not_rejected() -> None:
    # A flat (terrain=None) create_world must NOT hit the reject path; it falls
    # through to the real build (which __new__ cannot run), so we only assert
    # the reject branch does not fire by patching _rebuild/_lock to no-ops.
    import threading

    assert _engine_cls is not None
    eng = _engine_cls.__new__(_engine_cls)
    eng._lock = threading.RLock()
    eng.default_timestep = 0.002
    eng._solver_name = "mujoco"
    eng._rebuild = lambda: None  # type: ignore[method-assign]
    r = eng.create_world(terrain=None)
    assert r["status"] == "success"
