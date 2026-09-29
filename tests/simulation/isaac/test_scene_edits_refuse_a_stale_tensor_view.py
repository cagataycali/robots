"""Every call that reads PhysX's tensor view answers a stale view with ``reset()``, not a crash.

A dynamic ``remove_object`` or a ``remove_robot`` invalidates the tensor view
PhysX built at ``world.reset()``. The clock-advancing verbs already refused that
state (:mod:`tests.simulation.isaac.test_every_tick_refuses_a_scene_the_tensor_view_no_longer_covers`);
these did not, measured on one L40S with Isaac Sim 6.1:

* ``add_robot`` initialized the new articulation inside the dead view and failed
  with ``'NoneType' object has no attribute 'link_names'``. The failure path
  rebuilt enough state that the same call then worked, so remove/add cycles
  alternated error/success.
* ``get_body_state`` on a dynamic object, ``move_object`` on one and
  ``set_robot_pose`` raised a bare ``Exception`` ("Failed to get rigid body
  transforms from backend" / "... root link transforms ...") straight through
  the tool envelope.
* ``get_jacobian`` read the view with nothing checking it.

``get_body_state`` of a dynamic object no longer touches the dead handle: it
reads the pose off the USD stage (what the renderer draws), and with no stage
to read it names the stale view and ``reset()``.

Unit-level, like the sibling modules: the Kit leaves are stood in by handles
that raise the bare ``Exception`` the tensor API raises, so what is graded is
which paths refuse and what they say.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("strands_robots.simulation.isaac")

from strands_robots.simulation.isaac.simulation import _ObjectState, _RobotState  # noqa: E402
from tests.simulation._isaac_engine import isaac_engine  # noqa: E402

_REMEDY = "reset()"


class _DeadView(Exception):
    """What ``omni.physics.tensors`` raises on an invalidated view: a bare ``Exception``."""


class _RigidHandle:
    """A rigid-prim handle over an invalidated view: every read raises."""

    def get_world_pose(self) -> Any:
        raise _DeadView("Failed to get rigid body transforms from backend")

    def set_world_pose(self, *a: Any, **k: Any) -> None:
        raise _DeadView("Failed to get rigid body transforms from backend")

    def get_linear_velocity(self) -> Any:
        raise _DeadView("Failed to get rigid body velocities from backend")


class _PlainPrim:
    """A handle over a live view (or a static prim): reads and writes work."""

    moved_to: list[float] | None = None

    def get_world_pose(self) -> Any:
        return [0.3, 0.0, 0.05], [1.0, 0.0, 0.0, 0.0]

    def set_world_pose(self, position: Any = None, orientation: Any = None) -> None:
        self.moved_to = None if position is None else [float(v) for v in position]


class _Articulation:
    dof_names = ["j0"]

    def set_world_pose(self, *a: Any, **k: Any) -> None:
        raise _DeadView("Failed to get root link transforms from backend")

    def get_world_pose(self) -> Any:
        raise _DeadView("Failed to get root link transforms from backend")


class _World:
    physics_sim_view = object()


def _engine(*, stale: bool) -> Any:
    engine = isaac_engine()
    engine._world = _World()
    engine._world_created = True
    engine._physics_view_stale = stale
    robot = _RobotState(name="arm", prim_path="/World/Robots/arm", joint_names=["j0"])
    robot.articulation = _Articulation()
    engine._robots = {"arm": robot}
    engine._objects = {
        "cube": _ObjectState(
            name="cube", prim_path="/World/Objects/cube", shape="box", is_static=False, handle=_RigidHandle()
        )
    }
    return engine


def _text(result: dict[str, Any]) -> str:
    return " ".join(block.get("text", "") for block in result.get("content", []))


_CALLS = {
    "add_robot": lambda e: e.add_robot("arm_b", data_config="so101"),
    "set_robot_pose": lambda e: e.set_robot_pose("arm", position=[0.1, 0.0, 0.0]),
    "move_object": lambda e: e.move_object("cube", position=[0.3, 0.1, 0.05]),
    "get_jacobian": lambda e: e.get_jacobian(body_name="j0", robot_name="arm"),
}


class TestAStaleViewIsAnsweredNotRaised:
    @pytest.mark.parametrize("verb", sorted(_CALLS))
    def test_the_call_returns_the_refusal_naming_itself_and_reset(self, verb: str) -> None:
        result = _CALLS[verb](_engine(stale=True))
        assert result["status"] == "error"
        text = _text(result)
        assert verb in text and _REMEDY in text and "tensor view" in text

    def test_add_robot_after_remove_robot_says_reset_comes_first(self) -> None:
        text = _text(_CALLS["add_robot"](_engine(stale=True)))
        assert "before the next add_robot" in text

    def test_get_body_state_of_a_dynamic_object_names_the_stale_view(self) -> None:
        result = _engine(stale=True).get_body_state("cube")
        assert result["status"] == "error"
        text = _text(result)
        assert "added or removed since the last reset()" in text and "call reset()" in text

    def test_a_static_object_still_moves(self) -> None:
        engine = _engine(stale=True)
        engine._objects["cube"].is_static = True
        engine._objects["cube"].handle = _PlainPrim()
        # The USD half of the write needs Kit, absent here; the pose write through
        # the handle comes first, and reaching it is what shows no refusal fired.
        try:
            assert "tensor view" not in _text(engine.move_object("cube", position=[0.3, 0.1, 0.05]))
        except ImportError:
            pass
        assert engine._objects["cube"].handle.moved_to == [0.3, 0.1, 0.05]


class TestALiveViewIsNotRefused:
    @pytest.mark.parametrize("verb", sorted(set(_CALLS) - {"add_robot"}))
    def test_the_guard_does_not_fire(self, verb: str) -> None:
        # The stand-in handles raise, so these calls cannot succeed here; what
        # matters is that they are not answered with the stale-view refusal.
        try:
            text = _text(_CALLS[verb](_engine(stale=False)))
        except _DeadView:
            return
        assert "tensor view" not in text

    def test_get_body_state_does_not_blame_a_live_view(self) -> None:
        engine = _engine(stale=False)
        engine._objects["cube"].handle = _PlainPrim()
        result = engine.get_body_state("cube")
        assert result["status"] == "success" and "added or removed" not in _text(result)
