"""One pose-vector rule for every entry point that takes a pose.

``add_object`` / ``move_object`` / ``add_camera`` / ``add_robot`` route their
position/orientation/target through the shared pose-vector guard. ``move_to``
- the motion primitive that takes the same ``[x, y, z]`` and the same wxyz
quaternion - hand-rolled its own check, and the two contracts had drifted apart
in both directions:

* ``move_to`` computed ``len(position)`` directly, so a value with no length
  (a scalar, a NumPy 0-d scalar, a generator from ``map``/a comprehension)
  raised a bare ``TypeError: object of type 'float' has no len()`` straight
  through the ``{"status": "error"}`` contract its own docstring promises it
  never breaks. The scene calls refuse the same value cleanly.
* The scene calls accepted a ``bool`` component, because ``float(True)`` is
  ``1.0``: ``add_object(position=[True, 0, 0.3])`` reported success and placed
  the body a metre out on x. ``move_to`` refused it, and so does the agent-tool
  router - so the direct API was the only surface that took it.
* The quaternion domain drifted the other way. ``move_to`` refused an
  orientation whose norm rounds to zero - a value that describes no rotation -
  with a hand-rolled check sitting right after the shared pose guard, and the
  scene calls accepted it: ``move_object(orientation=[0, 0, 0, 0])`` reported
  success and echoed that quaternion back while ``get_body_state`` reported
  identity, because MuJoCo substitutes identity for a zero body quaternion
  without complaint. Its own XML door refuses the same value outright.

These pin the unified contract: every entry point refuses a pose it cannot
read, refuses a ``bool`` where a coordinate belongs, accepts a NumPy array, and
- for ``move_to`` - an orientation that is supplied genuinely constrains the IK
solve rather than being validated and dropped.
"""

import numpy as np
import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

from .test_motion_primitives import ARM_XML, REACHABLE  # noqa: E402

# Values with no length. Each one raised ``TypeError`` out of ``move_to``'s
# ``len(position)`` before the shared guard was applied; the scene entry points
# have always refused them.
UNREADABLE_POSES = [
    pytest.param(0.5, id="python-scalar"),
    pytest.param(3, id="python-int"),
    pytest.param(np.float64(1.0), id="numpy-scalar"),
    pytest.param((component for component in (0.2, 0.1, 0.2)), id="generator"),
]


#: A wxyz value with four finite components and no direction. Every entry
#: point below is asked to apply it.
ZERO_QUATERNION = [0.0, 0.0, 0.0, 0.0]


def _text(result):
    return " ".join(block["text"] for block in result["content"] if "text" in block)


def _json(result):
    return next(block["json"] for block in result["content"] if "json" in block)


@pytest.fixture
def arm_path(tmp_path):
    path = tmp_path / "pose_rule_arm.xml"
    path.write_text(ARM_XML)
    return str(path)


@pytest.fixture
def sim(arm_path):
    s = Simulation(tool_name="test_pose_vector_rule_parity", mesh=False)
    assert s.create_world(gravity=[0, 0, 0])["status"] == "success"
    assert s.add_robot("arm", urdf_path=arm_path)["status"] == "success"
    assert s.add_object(name="crate", shape="box", size=[0.05] * 3, position=[0.0, 0.0, 0.3])["status"] == "success"
    yield s
    s.cleanup(policy_stop_timeout=2.0)


class TestMoveToRefusesAPoseItCannotRead:
    """A pose with no length is a structured error, never a raise."""

    @pytest.mark.parametrize("bad", UNREADABLE_POSES)
    def test_position_without_a_length_is_refused(self, sim, bad):
        result = sim.move_to(robot_name="arm", position=bad, max_steps=2)
        assert result["status"] == "error"
        assert "move_to" in _text(result) and "position" in _text(result)

    @pytest.mark.parametrize("bad", UNREADABLE_POSES)
    def test_orientation_without_a_length_is_refused(self, sim, bad):
        result = sim.move_to(robot_name="arm", position=REACHABLE, orientation=bad, max_steps=2)
        assert result["status"] == "error"
        assert "move_to" in _text(result) and "orientation" in _text(result)

    def test_wrong_component_count_names_the_count(self, sim):
        result = sim.move_to(robot_name="arm", position=[0.2, 0.1], max_steps=2)
        assert result["status"] == "error"
        assert "3-element" in _text(result)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_non_finite_component_is_refused(self, sim, bad):
        result = sim.move_to(robot_name="arm", position=[0.2, bad, 0.2], max_steps=2)
        assert result["status"] == "error"
        assert "finite" in _text(result)


class TestVerdictsMatchAcrossEntryPoints:
    """``move_to`` and the scene calls accept and refuse the same poses."""

    # Factories, not values: a generator is consumed by whichever entry point
    # reads it first, so each side of the comparison needs its own.
    @pytest.mark.parametrize(
        "make_pose",
        [
            pytest.param(lambda: 0.5, id="python-scalar"),
            pytest.param(lambda: np.float64(1.0), id="numpy-scalar"),
            pytest.param(lambda: (c for c in (0.2, 0.1, 0.2)), id="generator"),
            pytest.param(lambda: [0.2, 0.1], id="too-short"),
            pytest.param(lambda: [0.2, float("nan"), 0.2], id="nan-component"),
            pytest.param(lambda: [True, 0.1, 0.2], id="bool-component"),
        ],
    )
    def test_a_pose_one_entry_point_refuses_is_refused_by_the_other(self, sim, make_pose):
        primitive = sim.move_to(robot_name="arm", position=make_pose(), max_steps=2)
        scene = sim.move_object(name="crate", position=make_pose())
        assert primitive["status"] == scene["status"] == "error", (
            f"verdicts differ: move_to={primitive['status']}, move_object={scene['status']}"
        )

    def test_a_numpy_array_pose_is_accepted_by_both(self, sim):
        pose = np.array([0.2, 0.1, 0.2])
        assert sim.move_object(name="crate", position=pose)["status"] == "success"
        assert sim.move_to(robot_name="arm", position=pose, tol=0.02, max_steps=400)["status"] == "success"


class TestBoolIsNotACoordinate:
    """``float(True) == 1.0``, so a bool component silently placed a body 1 m out."""

    def test_add_object_refuses_a_bool_component(self, sim):
        result = sim.add_object(name="flagged", shape="box", size=[0.05] * 3, position=[True, 0.0, 0.3])
        assert result["status"] == "error"
        assert "elements must be numbers" in _text(result)
        assert "flagged" not in _text(sim.list_objects())

    def test_move_object_refuses_a_bool_and_leaves_the_pose_alone(self, sim):
        before = _json(sim.get_body_state("crate"))["position"]
        assert sim.move_object(name="crate", position=[True, 0.0, 0.3])["status"] == "error"
        assert _json(sim.get_body_state("crate"))["position"] == pytest.approx(before)

    def test_add_camera_refuses_a_bool_component(self, sim):
        assert sim.add_camera(name="cam", position=[True, 1.0, 1.0], target=[0, 0, 0])["status"] == "error"

    def test_add_robot_refuses_a_bool_component(self, sim, arm_path):
        assert sim.add_robot("second", urdf_path=arm_path, position=[True, 0.0, 0.0])["status"] == "error"

    def test_move_to_refuses_a_bool_component(self, sim):
        assert sim.move_to(robot_name="arm", position=[True, 0.0, 0.3], max_steps=2)["status"] == "error"

    def test_the_agent_tool_router_and_the_direct_call_agree(self, sim):
        fields = {"action": "move_object", "name": "crate", "position": [True, 0.0, 0.3]}
        routed = sim._dispatch_action("move_object", fields)
        direct = sim.move_object(name="crate", position=[True, 0.0, 0.3])
        assert routed["status"] == direct["status"] == "error"


class TestOrientationEntersTheSolve:
    """A supplied quaternion constrains IK; it is normalized, not just checked."""

    def test_a_non_unit_quaternion_is_normalized(self, sim):
        pytest.importorskip("mink")
        unit = sim.move_to(robot_name="arm", position=REACHABLE, orientation=[0.0, 1.0, 0.0, 0.0], max_steps=2)
        scaled = sim.move_to(robot_name="arm", position=REACHABLE, orientation=[0.0, 2.0, 0.0, 0.0], max_steps=2)
        assert _json(unit)["ik_residual_m"] == pytest.approx(_json(scaled)["ik_residual_m"], abs=1e-9)

    def test_a_different_quaternion_changes_the_solve(self, sim):
        pytest.importorskip("mink")
        flipped = sim.move_to(robot_name="arm", position=REACHABLE, orientation=[0.0, 1.0, 0.0, 0.0], max_steps=2)
        quarter = sim.move_to(robot_name="arm", position=REACHABLE, orientation=[0.7071, 0.0, 0.7071, 0.0], max_steps=2)
        assert _json(flipped)["ik_residual_m"] != pytest.approx(_json(quarter)["ik_residual_m"], abs=1e-4)

    def test_position_only_reaches_a_target_the_full_pose_cannot(self, sim):
        pytest.importorskip("mink")
        # The fixture arm has 4 arm DOF, so an arbitrary full pose is not
        # realizable - the residual proves the orientation is a real constraint
        # rather than a validated-then-discarded parameter.
        assert sim.move_to(robot_name="arm", position=REACHABLE, tol=0.02, max_steps=400)["status"] == "success"
        constrained = sim.move_to(
            robot_name="arm", position=REACHABLE, orientation=[0.0, 1.0, 0.0, 0.0], tol=0.02, max_steps=400
        )
        assert constrained["status"] == "error"
        assert _json(constrained)["ik_residual_m"] > 0.02

    def test_a_zero_norm_quaternion_is_refused(self, sim):
        result = sim.move_to(robot_name="arm", position=REACHABLE, orientation=[0.0, 0.0, 0.0, 0.0], max_steps=2)
        assert result["status"] == "error"
        assert "zero norm" in _text(result)


class TestEveryEntryPointRefusesAQuaternionWithNoDirection:
    """A zero-norm wxyz value is no rotation, whichever surface receives it."""

    def test_mujoco_refuses_it_through_its_xml_door(self):
        """The oracle: the same value, offered to MuJoCo as MJCF, is refused."""
        import mujoco

        xml = '<mujoco><worldbody><body name="b" quat="0 0 0 0">'
        xml += '<geom type="box" size=".1 .1 .1"/></body></worldbody></mujoco>'
        with pytest.raises(ValueError, match="zero quaternion"):
            mujoco.MjModel.from_xml_string(xml)

    def test_move_object_refuses_it_and_leaves_the_orientation_alone(self, sim):
        quarter_turn = [0.7071, 0.0, 0.7071, 0.0]
        assert sim.move_object(name="crate", orientation=quarter_turn)["status"] == "success"
        before = _json(sim.get_body_state("crate"))["quaternion"]

        refused = sim.move_object(name="crate", orientation=ZERO_QUATERNION)
        assert refused["status"] == "error"
        assert "zero norm" in _text(refused)
        # The old success text named the zero quaternion while the body carried
        # identity; the pose the body does have is the pose it had before.
        assert _json(sim.get_body_state("crate"))["quaternion"] == pytest.approx(before)

    def test_add_object_refuses_it_and_adds_nothing(self, sim):
        refused = sim.add_object(
            name="spun", shape="box", size=[0.05] * 3, position=[0.3, 0.0, 0.3], orientation=ZERO_QUATERNION
        )
        assert refused["status"] == "error"
        assert "zero norm" in _text(refused)
        assert "spun" not in _text(sim.list_objects())

    def test_add_robot_refuses_it(self, sim, arm_path):
        refused = sim.add_robot("spun_arm", urdf_path=arm_path, orientation=ZERO_QUATERNION)
        assert refused["status"] == "error"
        assert "zero norm" in _text(refused)

    @pytest.mark.parametrize(
        "op",
        [
            pytest.param({"op": "set_body_quat", "name": "crate"}, id="set_body_quat"),
            pytest.param({"op": "add_body", "name": "spun_body"}, id="add_body"),
        ],
    )
    def test_a_structured_scene_op_refuses_it(self, sim, op):
        refused = sim.patch_scene_mjcf(ops=[{**op, "quat": ZERO_QUATERNION}])
        assert refused["status"] == "error"
        assert "zero norm" in _text(refused)

    def test_the_scene_verdict_matches_move_to(self, sim):
        """The parity this module exists for, on the orientation parameter."""
        primitive = sim.move_to(robot_name="arm", position=REACHABLE, orientation=ZERO_QUATERNION, max_steps=2)
        scene = sim.move_object(name="crate", orientation=ZERO_QUATERNION)
        assert primitive["status"] == scene["status"] == "error", (
            f"verdicts differ: move_to={primitive['status']}, move_object={scene['status']}"
        )


class TestMagnitudeIsStillNotPartOfTheContract:
    """Only a norm with no direction is refused; any other magnitude is fine."""

    def test_a_non_unit_quaternion_is_applied_and_reported_normalized(self, sim):
        assert sim.move_object(name="crate", orientation=[0.0, 2.0, 0.0, 0.0])["status"] == "success"
        assert _json(sim.get_body_state("crate"))["quaternion"] == pytest.approx([0.0, 1.0, 0.0, 0.0])

    def test_a_scene_op_accepts_a_non_unit_quaternion(self, sim):
        result = sim.patch_scene_mjcf(ops=[{"op": "set_body_quat", "name": "crate", "quat": [0.0, 2.0, 0.0, 0.0]}])
        assert result["status"] == "success"
