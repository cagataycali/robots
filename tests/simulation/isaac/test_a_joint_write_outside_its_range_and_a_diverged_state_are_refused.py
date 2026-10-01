"""Isaac refuses a joint write outside the joint's range, and stops reporting success on NaN.

Measured on one L40S (Isaac Sim 6.1, so100):

* ``set_joint_positions({"Elbow": 50})`` - 50 degrees, sent as radians - was
  "Set joint positions (main)." and 60 steps later every joint was NaN;
  ``{"Rotation": 2.5}`` on a [-1.92, 1.92] joint read back 2.5, then snapped to
  1.82 and kicked Wrist_Roll from 0.02 to 1.17 rad. MuJoCo refuses both, and
  writes nothing.
* on the NaN state, ``step`` kept answering "Stepped 1x ... success" and
  ``send_action`` "Action applied", so a rollout ran on for its whole horizon on
  a robot that was no longer being simulated.

Unit-level: the articulation is a stand-in with limits and a settable state.
"""

from __future__ import annotations

import types
from typing import Any

import numpy as np
import pytest

pytest.importorskip("strands_robots.simulation.isaac")

from strands_robots.simulation.isaac.simulation import (  # noqa: E402
    IsaacConfig,
    _diverged_robots_error,
    _dof_units,
    _RobotState,
)
from tests.simulation._isaac_engine import isaac_engine  # noqa: E402

_JOINTS = ["Rotation", "Pitch", "Elbow", "Jaw"]


class _Articulation:
    def __init__(self, q: list[float] | None = None) -> None:
        self.q = np.array(q if q is not None else [0.0, 0.0, 0.0, 0.0])
        self.dof_properties = {
            "lower": np.array([-1.92, -3.32, -0.174, -0.174]),
            "upper": np.array([1.92, 0.174, 3.14, 1.75]),
            "hasLimits": np.array([True, True, True, True]),
            "type": np.array([1, 1, 1, 2]),
        }
        self.written: list[Any] = []

    def get_joint_positions(self) -> np.ndarray:
        return self.q

    def set_joint_positions(self, *args: Any, **kwargs: Any) -> None:
        self.written.append((args, kwargs))


def _engine(q: list[float] | None = None) -> Any:
    engine: Any = isaac_engine(IsaacConfig(headless=True))
    engine._world = types.SimpleNamespace()
    engine._world_created = True
    engine._physics_view_stale = False
    robot = _RobotState(name="arm", prim_path="/World/Robots/arm", joint_names=list(_JOINTS))
    robot.articulation = _Articulation(q)
    engine._robots = {"arm": robot}
    return engine


def _text(result: dict[str, Any]) -> str:
    return " ".join(b.get("text", "") for b in result.get("content", []))


class TestAWriteOutsideTheRangeIsRefused:
    @pytest.mark.parametrize(
        ("positions", "needle"),
        [
            ({"Elbow": 50.0}, "Elbow=50 outside [-0.174, 3.14] rad"),
            ({"Rotation": 2.5}, "Rotation=2.5 outside [-1.92, 1.92] rad"),
            ([0.0, 0.0, 0.0, 1.9], "Jaw=1.9 outside [-0.174, 1.75] m"),
        ],
    )
    def test_it_names_the_joint_and_its_range_and_writes_nothing(self, positions: Any, needle: str) -> None:
        engine = _engine()
        result = engine.set_joint_positions(positions, robot_name="arm")
        assert result["status"] == "error"
        assert "nothing written" in _text(result) and needle in _text(result)
        assert engine._robots["arm"].articulation.written == []

    def test_degrees_are_named_when_the_value_would_fit_in_radians(self) -> None:
        text = _text(_engine().set_joint_positions({"Elbow": 50.0}, robot_name="arm"))
        assert "radians, not degrees: 50 deg = 0.8727 rad" in text

    def test_a_prismatic_joint_gets_no_degree_hint(self) -> None:
        text = _text(_engine().set_joint_positions({"Jaw": 1.9}, robot_name="arm"))
        assert "degrees" not in text

    def test_a_value_inside_the_range_is_not_refused(self) -> None:
        engine = _engine()
        assert engine._joint_range_error(engine._robots["arm"], _JOINTS, {"Elbow": 1.0, "Rotation": -1.9}) is None

    def test_a_dof_without_limits_is_not_checked(self) -> None:
        engine = _engine()
        engine._robots["arm"].articulation.dof_properties["hasLimits"] = np.array([False] * 4)
        assert engine._joint_range_error(engine._robots["arm"], _JOINTS, {"Elbow": 50.0}) is None


class TestADivergedStateIsNotReportedAsSuccess:
    def test_nan_joints_are_named_with_the_remedy(self) -> None:
        engine = _engine([0.1, float("nan"), float("nan"), 0.0])
        result = _diverged_robots_error(engine, "step")
        assert result is not None and result["status"] == "error"
        text = _text(result)
        assert "'arm' (Pitch, Elbow)" in text and "reset()" in text and text.startswith("step: the physics diverged")

    def test_finite_joints_are_fine(self) -> None:
        assert _diverged_robots_error(_engine([0.1, 0.2, 0.3, 0.0]), "step") is None

    def test_a_stale_view_is_not_read(self) -> None:
        engine = _engine([float("nan")] * 4)
        engine._physics_view_stale = True
        assert _diverged_robots_error(engine, "send_action") is None

    def test_the_stale_check_and_the_reads_happen_under_the_engine_lock(self) -> None:
        """A worker's remove_object between the check and the read would leave a stale view being read (#4076).

        The callers release ``_lock`` after their last batch, so the helper takes
        it itself: the stale flag is read and every articulation read happens
        while the lock is held, as one step.
        """
        import threading

        engine = _engine([0.1, float("nan"), 0.0, 0.0])
        engine._lock = threading.RLock()
        seen: list[bool] = []

        class _LockAwareArticulation(_Articulation):
            def get_joint_positions(self) -> Any:
                # RLock.acquire(blocking=False) from ANOTHER thread fails iff this thread holds it.
                probe: list[bool] = []

                def _try() -> None:
                    got = engine._lock.acquire(blocking=False)
                    probe.append(got)
                    if got:
                        engine._lock.release()

                t = threading.Thread(target=_try)
                t.start()
                t.join()
                seen.append(not probe[0])
                return super().get_joint_positions()

        engine._robots["arm"].articulation = _LockAwareArticulation([0.1, float("nan"), 0.0, 0.0])
        result = _diverged_robots_error(engine, "step")
        assert result is not None and "'arm' (Pitch)" in _text(result)
        assert seen == [True], "the articulation was read while the engine lock was held"
        assert engine._lock.acquire(blocking=False), "and the lock is released afterwards"
        engine._lock.release()

    def test_an_engine_without_a_lock_is_still_checked(self) -> None:
        engine = _engine([float("nan"), 0.0, 0.0, 0.0])
        engine._lock = None
        result = _diverged_robots_error(engine, "step")
        assert result is not None and "Rotation" in _text(result)

    def test_step_returns_it(self) -> None:
        engine = _engine([float("inf"), 0.0, 0.0, 0.0])
        engine._world = types.SimpleNamespace(step=lambda render=False: None, current_time=0.0)
        engine._sim_time, engine._step_count = 0.0, 0
        result = engine.step(1)
        assert result["status"] == "error" and "physics diverged" in _text(result)


class TestTheDofUnit:
    def test_rotation_and_translation(self) -> None:
        assert _dof_units(_Articulation(), 4) == ["rad", "rad", "rad", "m"]

    def test_unknown_without_the_field(self) -> None:
        assert _dof_units(types.SimpleNamespace(), 2) == ["", ""]
