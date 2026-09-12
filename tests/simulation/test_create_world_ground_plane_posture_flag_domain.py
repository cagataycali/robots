# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``create_world`` checks ``ground_plane`` instead of reading it by truthiness.

``ground_plane`` selects one of two postures for the world being built: lay a
floor at ``z=0`` or leave the world open. Both backends that build a world
validated every knob beside it - ``timestep`` and ``gravity`` through the shared
numeric bindings, ``terrain`` and ``difficulty`` through their own domains - and
read this one by truthiness: MuJoCo compiled the plane ``if world.ground_plane``
and Newton called ``builder.add_ground_plane()`` under the same test. Every
non-empty string is truthy, so the spellings a caller reaches for to decline the
floor laid it, and the falsy non-booleans omitted it without being a declared
spelling of that. Measured on ``main`` at ``07ba807`` through the MuJoCo facade,
reading ``geom_type`` off the compiled model:

| ``create_world(ground_plane=)`` | status | plane geom compiled |
|---|---|---|
| ``True`` / ``False`` | success | as asked |
| ``"false"`` ``"no"`` ``"off"`` ``"0"`` ``1`` ``nan`` | success | **yes** - the floor the word declines |
| ``None`` ``0`` ``0.0`` ``""`` ``[]`` ``{}`` | success | no - without being a spelling of "no" |

Nothing raised and nothing logged on either half. What a caller sees is a
scene: a floating-base robot spawned into a world it asked to be open lands on
a floor, and a policy evaluated on a world it asked to have a floor falls
through the one it did not get, reported by whatever predicate happens to read
the height first.

The fix binds the shared :func:`~strands_robots.utils.boolean_flag_error`
domain to each backend's ``create_world`` through the
``SimEngine._validate_posture_flags`` envelope the rollout facades already use.
It sits after the ``terrain`` / ``difficulty`` domain guards - the flag gates
neither of those reads, and ``tests/simulation/test_create_world_difficulty_domain.py``
drives both backends on a stub carrying only what runs before *its* guard - and
ahead of everything that reads the flag: MuJoCo's world-exists report, which
describes the world it cannot rebuild in terms of it, and the build itself. The
refused call builds nothing - the same ``Simulation`` then creates a world
normally.

The Isaac backend reads the same flag by truthiness and is out of scope here:
its ``create_world`` is rewritten by an open pull request against
``strands_robots/simulation/isaac/simulation.py``, so it is a merge-order item
for after that lands.
"""

from __future__ import annotations

import math
import threading
from typing import TYPE_CHECKING, Any

import pytest

from strands_robots.utils import boolean_flag_error

if TYPE_CHECKING:
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

#: Truthy spellings of "no floor": each laid the floor the word declines.
TRUTHY_NON_BOOLEANS: tuple[Any, ...] = ("false", "no", "off", "0", 1, 2, math.nan)
#: Falsy non-booleans: each omitted the floor without being a spelling of "no".
FALSY_NON_BOOLEANS: tuple[Any, ...] = (None, 0, 0.0, "", [], {})
UNDECLARED: tuple[Any, ...] = (*TRUTHY_NON_BOOLEANS, *FALSY_NON_BOOLEANS)


def _text(result: dict[str, Any]) -> str:
    return "".join(block.get("text", "") for block in result["content"])


def _plane_compiled(sim: Any) -> bool:
    """Whether the compiled MuJoCo model carries a plane geom.

    Compared ``int()`` to ``int()``: ``geom_type`` is a NumPy integer array and
    ``mjGEOM_PLANE`` a pybind11 enum, and membership puts the enum on the left
    (AGENTS.md > MuJoCo enums are matched by value).
    """
    import mujoco

    model = sim._world._model
    plane = int(mujoco.mjtGeom.mjGEOM_PLANE)
    return any(int(model.geom_type[i]) == plane for i in range(model.ngeom))


class TestMuJoCoCreateWorldChecksGroundPlane:
    """The MuJoCo backend refuses a non-boolean ``ground_plane`` and builds nothing."""

    @pytest.fixture
    def sim(self) -> Any:
        pytest.importorskip("mujoco")
        from strands_robots import Simulation

        sim = Simulation(backend="mujoco", tool_name="ground_plane_domain_test", mesh=False)
        yield sim
        sim.cleanup()

    @pytest.mark.parametrize("value", UNDECLARED, ids=repr)
    def test_an_undeclared_spelling_is_refused_by_name(self, sim: Any, value: Any) -> None:
        result = sim.create_world(ground_plane=value)
        assert result["status"] == "error"
        assert _text(result) == boolean_flag_error(value, "ground_plane", "create_world")

    @pytest.mark.parametrize("value", UNDECLARED, ids=repr)
    def test_the_refused_call_builds_no_world(self, sim: Any, value: Any) -> None:
        sim.create_world(ground_plane=value)
        # A world left behind by the refusal would make this the world-exists
        # report rather than a fresh build.
        result = sim.create_world()
        assert result["status"] == "success", _text(result)
        assert _plane_compiled(sim)

    def test_the_flag_is_refused_ahead_of_the_report_that_reads_it(self, sim: Any) -> None:
        # The world-exists report describes the world it cannot rebuild in
        # terms of ``ground_plane``, so a misread posture is named before that
        # report can render it.
        assert sim.create_world()["status"] == "success"
        result = sim.create_world(ground_plane="false")
        assert result["status"] == "error"
        assert _text(result) == boolean_flag_error("false", "ground_plane", "create_world")

    def test_an_unusable_terrain_is_still_named_first(self, sim: Any) -> None:
        # The flag gates neither ``terrain`` nor ``difficulty``, so their
        # domains keep their place ahead of it.
        result = sim.create_world(ground_plane="false", terrain="not-a-terrain")
        assert result["status"] == "error"
        assert "not-a-terrain" in _text(result)
        assert "ground_plane" not in _text(result)

    def test_true_still_lays_the_floor(self, sim: Any) -> None:
        result = sim.create_world(ground_plane=True)
        assert result["status"] == "success", _text(result)
        assert _plane_compiled(sim)

    def test_false_still_leaves_the_world_open(self, sim: Any) -> None:
        result = sim.create_world(ground_plane=False)
        assert result["status"] == "success", _text(result)
        assert not _plane_compiled(sim)

    def test_the_world_exists_report_is_unchanged_for_a_boolean(self, sim: Any) -> None:
        assert sim.create_world(ground_plane=True)["status"] == "success"
        result = sim.create_world(ground_plane=False)
        assert result["status"] == "error"
        assert "ground_plane=False" in _text(result)


_newton_engine_cls: type[NewtonSimEngine] | None
try:  # NewtonSimEngine imports without Warp (Warp is lazily loaded at build time)
    from strands_robots.simulation.newton.simulation import NewtonSimEngine as _newton_engine_cls
except Exception:  # pragma: no cover - newton package genuinely absent
    _newton_engine_cls = None


@pytest.mark.skipif(_newton_engine_cls is None, reason="newton package not importable")
class TestNewtonCreateWorldChecksGroundPlane:
    """The Newton backend refuses the same spellings before any build."""

    @pytest.fixture
    def engine(self) -> Any:
        # ``__new__`` bypasses ``__init__`` (no solver, no GPU); the refusal
        # returns before the lock and the rebuild, so a refused call needs no
        # engine state and a control build only needs the two it reads.
        assert _newton_engine_cls is not None
        eng = _newton_engine_cls.__new__(_newton_engine_cls)
        eng._lock = threading.RLock()
        eng.default_timestep = 0.002
        eng._solver_name = "mujoco"
        eng._rebuild = lambda: None  # type: ignore[method-assign]
        return eng

    @pytest.mark.parametrize("value", UNDECLARED, ids=repr)
    def test_an_undeclared_spelling_is_refused_by_name(self, engine: Any, value: Any) -> None:
        result = engine.create_world(ground_plane=value)
        assert result["status"] == "error"
        assert _text(result) == boolean_flag_error(value, "ground_plane", "create_world")
        assert getattr(engine, "_world", None) is None, "the refused call built a world"

    def test_an_unsupported_terrain_is_still_named_first(self, engine: Any) -> None:
        result = engine.create_world(ground_plane="false", terrain="rough")
        assert result["status"] == "error"
        assert "rough" in _text(result)
        assert "ground_plane" not in _text(result)

    def test_the_flag_is_refused_ahead_of_the_timestep_it_would_build_with(self, engine: Any) -> None:
        # Both are refusals; the posture is named first so a caller who
        # mistyped the flag is not sent to correct a timestep instead.
        result = engine.create_world(ground_plane="false", timestep=-1.0)
        assert result["status"] == "error"
        assert _text(result) == boolean_flag_error("false", "ground_plane", "create_world")

    @pytest.mark.parametrize("value", (True, False), ids=repr)
    def test_a_boolean_reaches_the_world_as_given(self, engine: Any, value: bool) -> None:
        result = engine.create_world(ground_plane=value)
        assert result["status"] == "success", _text(result)
        assert engine._world.ground_plane is value
