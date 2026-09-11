"""The Newton backend's scene listings report measured poses, not requests.

``SimRobot.position`` and ``SimObject.position`` hold the vector ``add_robot``
/ ``add_object`` was *asked* for. Newton never writes either one back, so both
listings described the scene by its spawn arguments: wrong the moment a body
moves, and wrong at ``t=0`` for any model whose authored root pose is composed
with the requested transform rather than replaced by it.

Every cell grades the public listing text against the solver's own ``body_q``,
reached through arrays that predate this change (``_robot_body_map``,
``joint_parent`` / ``joint_child``, and - for an object - the body no robot
owns), so the oracle is independent of the code under test.

Needs the real Newton engine (a finalized model and a stepped solver state), so
the module is skipped when Newton/Warp are absent.
"""

from __future__ import annotations

import importlib.util

import pytest

_HAS_NEWTON = importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None

pytestmark = pytest.mark.skipif(not _HAS_NEWTON, reason="newton/warp not installed")


def _engine(robot: str | None = None, **robot_kwargs):
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

    sim = NewtonSimEngine(solver="mujoco")
    sim.create_world(ground_plane=True)
    if robot is not None:
        assert sim.add_robot(robot, **robot_kwargs)["status"] == "success"
    return sim


def _body_truth(sim, body_index: int) -> list[float]:
    """The solver's own world position for one body, to 4 decimal places."""
    return [round(float(v), 4) for v in sim._state_0.body_q.numpy()[body_index][:3]]


def _root_body_indices(sim, robot_name: str) -> list[int]:
    """Indices of ``robot_name``'s world-parented bodies, from pre-existing arrays."""
    labels = list(sim._model.body_label)
    owned = {labels.index(label) for label in sim._robot_body_map.get(robot_name, []) if label in labels}
    parents = sim._model.joint_parent.numpy()
    children = sim._model.joint_child.numpy()
    return [int(c) for p, c in zip(parents, children, strict=True) if int(p) == -1 and int(c) in owned]


def _sole_object_body_index(sim) -> int:
    """The body of the scene's one dynamic object, resolved without the engine's map.

    Newton labels an object's body positionally, so it is identified here by
    elimination: the only body that no robot contributed.
    """
    owned = {label for labels in sim._robot_body_map.values() for label in labels}
    free = [i for i, label in enumerate(sim._model.body_label) if label not in owned]
    assert len(free) == 1, f"expected exactly one object body, got {free}"
    return free[0]


def _listing_text(result: dict) -> str:
    assert result["status"] == "success", result
    return str(result["content"][0]["text"])


class TestRobotBaseIsMeasured:
    def test_single_root_base_is_the_live_body_not_the_requested_transform(self):
        """A go2 asked for z=0 stands at z=0.445: the request is wrong at t=0.

        ``position`` is the translation of the transform handed to ``add_mjcf``,
        and the model's authored root pose is composed with it rather than
        replaced, so the request never named where this robot stands.
        """
        sim = _engine("unitree_go2")
        try:
            roots = _root_body_indices(sim, "unitree_go2")
            assert len(roots) == 1
            truth = _body_truth(sim, roots[0])
            requested = list(sim._world.robots["unitree_go2"].position)
            assert truth != requested, "premise: this model's root carries an authored offset"
            assert f"Position: {truth}" in _listing_text(sim.list_robots_info())
        finally:
            sim.destroy()

    def test_base_tracks_the_robot_once_it_moves(self):
        sim = _engine("unitree_go2")
        try:
            roots = _root_body_indices(sim, "unitree_go2")
            before = _body_truth(sim, roots[0])
            sim.step(120)
            after = _body_truth(sim, roots[0])
            assert after != before, "premise: a settling floating base must have moved"
            assert f"Position: {after}" in _listing_text(sim.list_robots_info())
        finally:
            sim.destroy()

    def test_several_root_bodies_keep_the_request_and_say_so(self):
        """An aloha attaches two arm bases: no one base pose exists to measure."""
        sim = _engine("aloha", position=[2.0, 0.0, 0.0])
        try:
            assert len(_root_body_indices(sim, "aloha")) == 2
            text = _listing_text(sim.list_robots_info())
            assert "[2.0, 0.0, 0.0] (requested transform;" in text
            assert "no single root body" in text
        finally:
            sim.destroy()


class TestObjectPoseIsMeasured:
    def test_dynamic_object_pose_is_the_live_body_not_the_record(self):
        sim = _engine()
        try:
            requested = [0.4, 0.1, 0.5]
            assert (
                sim.add_object("cube", shape="box", position=list(requested), size=[0.05] * 3, mass=0.2)["status"]
                == "success"
            )
            sim.step(120)
            truth = _body_truth(sim, _sole_object_body_index(sim))
            assert truth != requested, "premise: gravity must have moved the cube"
            text = _listing_text(sim.list_objects())
            assert f"cube: box at {truth}" in text
            assert str(requested) not in text
        finally:
            sim.destroy()

    def test_a_moved_object_is_measured_from_the_rebuilt_model(self):
        """``move_object`` refinalizes the model, so the body must be re-resolved."""
        sim = _engine()
        try:
            sim.add_object("cube", shape="box", position=[0.4, 0.1, 0.5], size=[0.05] * 3, mass=0.2)
            assert sim.move_object("cube", position=[0.2, -0.3, 0.6])["status"] == "success"
            sim.step(120)
            truth = _body_truth(sim, _sole_object_body_index(sim))
            assert truth != [0.2, -0.3, 0.6], "premise: gravity must have moved the cube after the rebuild"
            assert f"cube: box at {truth}" in _listing_text(sim.list_objects())
        finally:
            sim.destroy()

    def test_static_object_keeps_its_record(self):
        """Newton gives a static shape no body of its own, and nothing moves it.

        The control: this line must read the same before and after the change.
        """
        sim = _engine()
        try:
            sim.add_object("pillar", shape="box", position=[-0.5, 0.0, 0.1], size=[0.05, 0.05, 0.1], is_static=True)
            sim.step(120)
            assert "pillar: box at [-0.5, 0.0, 0.1], static" in _listing_text(sim.list_objects())
        finally:
            sim.destroy()

    def test_dynamic_object_with_no_body_is_reported_not_guessed(self):
        sim = _engine()
        try:
            sim.add_object("cube", shape="box", position=[0.4, 0.1, 0.5], size=[0.05] * 3, mass=0.2)
            sim._object_body_map["cube"] = -1  # the record and the model diverged
            result = sim.list_objects()
            assert result["status"] == "error"
            assert "has no body in the finalized model" in result["content"][0]["text"]
        finally:
            sim.destroy()
