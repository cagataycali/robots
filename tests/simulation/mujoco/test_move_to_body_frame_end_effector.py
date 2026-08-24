"""``move_to`` on a body-framed end-effector.

``move_to`` does not take an EE frame: it auto-discovers one per robot namespace
with :func:`strands_robots.simulation.ik.discover_ee_frame`, whose three
documented outcomes its own docstring names - "TCP-like site, else hand/tool
body, else the chain's leaf body". Two of those three are ``"body"``, and the
frame is not an internal detail: the primitive reports ``frame`` /
``frame_type`` / ``ee_position`` / ``ee_orientation_wxyz`` in its json payload,
and measures ``position_error_m`` - the value the ``reached`` flag is decided
on - at that frame every control tick.

The discovery function is pinned exhaustively, all three branches, by
``tests/policies/vera/test_ee_frame_discovery.py``. Its consumer was not: every
arm in the motion-primitive suite declares an ``ee_site``, so ``move_to`` had
only ever run on the site branch, and the one ``frame_type`` assertion in the
tree pins ``"site"``. The body arm of the readback - both discovery routes that
reach it, the pose it reports and the convergence decision it drives - had never
executed.

The distinction that matters is which body frame is read. MuJoCo carries two per
body: the frame origin (``xpos``/``xquat``) and the inertial/CoM frame
(``xipos``), and on the arm below they are 25 mm apart. mink optimizes the frame
origin, so reading ``xipos`` would leave the solver and the convergence check
measuring points 25 mm apart - and with the documented default ``tol`` of 10 mm
a perfectly solved target could then never converge, reporting a servo timeout
for a pose that had already arrived. That agreement is pinned here directly.

The arm mirrors the shared ``ARM_XML`` kinematics with the TCP site and the jaw
removed, because a site-less model is what routes discovery to a body at all.
Its tip carries a ``fromto`` capsule, so the frame origin and the inertial frame
are genuinely distinct rather than coincident. Two names for one physical body
give the two body routes: ``hand`` matches a body hint, ``link4`` matches none
and is found as the chain tail.
"""

from typing import Any

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("mink")

from strands_robots.simulation.ik import discover_ee_frame  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

from .test_motion_primitives import ARM_XML, REACHABLE  # noqa: E402

# ARM_XML's kinematics without the TCP site and without the jaw. ``{tip}`` names
# the tool-mount body: "hand" matches ``_BODY_HINTS``, "link4" matches nothing
# and is reached as the kinematic chain's tail. The tip geom is a ``fromto``
# capsule, so its inertial frame sits 25 mm from the body origin.
_BODY_ARM_TEMPLATE = """
<mujoco model="body_frame_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 0"/>
  <default>
    <joint armature="0.05" damping="0.5"/>
    <geom density="2000"/>
  </default>
  <worldbody>
    <body name="base" pos="0 0 0">
      <geom type="cylinder" size="0.04 0.02"/>
      <body name="link1" pos="0 0 0.05">
        <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.02"/>
        <body name="link2" pos="0 0 0.2">
          <joint name="shoulder_lift" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
          <geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.02"/>
          <body name="link3" pos="0.15 0 0">
            <joint name="elbow" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
            <geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.018"/>
            <body name="{tip}" pos="0.15 0 0">
              <joint name="wrist_roll" type="hinge" axis="1 0 0" range="-3.0 3.0"/>
              <geom type="capsule" fromto="0 0 0 0.05 0 0" size="0.015"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_pan" joint="shoulder_pan" kp="20" dampratio="1" ctrlrange="-3.14 3.14"/>
    <position name="shoulder_lift" joint="shoulder_lift" kp="20" dampratio="1" ctrlrange="-3.14 3.14"/>
    <position name="elbow" joint="elbow" kp="20" dampratio="1" ctrlrange="-3.14 3.14"/>
    <position name="wrist_roll" joint="wrist_roll" kp="5" dampratio="1" ctrlrange="-3.0 3.0"/>
  </actuator>
</mujoco>
"""

# ``tol`` is looser than the residual this 4-DOF arm leaves on ``REACHABLE`` so
# ``move_to`` enters its servo loop instead of refusing the target up front.
_TOL = 0.02
_MAX_STEPS = 400


def _arm_xml(tip: str) -> str:
    """The site-less arm with ``tip`` naming its tool-mount body."""
    xml = _BODY_ARM_TEMPLATE.format(tip=tip)
    assert "<site" not in xml, "a TCP site would route discovery to the site branch"
    return xml


def _sim_with(tmp_path: Any, tip: str) -> Any:
    """A world holding the site-less arm under the ``arm/`` namespace."""
    path = tmp_path / f"{tip}_arm.xml"
    path.write_text(_arm_xml(tip))
    sim = Simulation(backend="mujoco", mesh=False)
    assert sim.create_world()["status"] == "success"
    added = sim.add_robot(name="arm", urdf_path=str(path))
    assert added["status"] == "success", added
    return sim


def _json_block(result: dict[str, Any]) -> dict[str, Any]:
    """The primitive's structured payload."""
    return next(c["json"] for c in result["content"] if "json" in c)


def _reach(sim: Any) -> dict[str, Any]:
    """Drive ``move_to`` to ``REACHABLE`` and return its payload."""
    result = sim.move_to(robot_name="arm", position=REACHABLE, tol=_TOL, max_steps=_MAX_STEPS)
    assert result["status"] == "success", result
    return _json_block(result)


@pytest.fixture
def hand_body_sim(tmp_path):
    """Arm whose tool-mount body matches a body hint (discovery route 2)."""
    sim = _sim_with(tmp_path, "hand")
    try:
        yield sim
    finally:
        sim.cleanup()


@pytest.fixture
def leaf_body_sim(tmp_path):
    """Arm whose tool-mount body matches no hint (discovery route 3)."""
    sim = _sim_with(tmp_path, "link4")
    try:
        yield sim
    finally:
        sim.cleanup()


class TestTheShippedArmOnlyEverExercisedTheSiteRoute:
    """Why the body arm of the readback had no coverage.

    Not a contract on ``move_to`` - a premise about the suite. If the shared arm
    stopped resolving a site these tests would be measuring the same route as
    every other primitive test, so the gap they close is asserted rather than
    assumed.
    """

    def test_the_shared_arm_resolves_a_site(self, tmp_path):
        sim = _sim_with(tmp_path, "hand")
        try:
            path = tmp_path / "shared_arm.xml"
            path.write_text(ARM_XML)
            assert sim.add_robot(name="shared", urdf_path=str(path))["status"] == "success"
            assert discover_ee_frame(sim._world._model, "shared/") == ("shared/ee_site", "site")
        finally:
            sim.cleanup()

    def test_the_body_framed_arm_declares_no_site(self, tmp_path):
        sim = _sim_with(tmp_path, "hand")
        try:
            assert int(sim._world._model.nsite) == 0
        finally:
            sim.cleanup()


class TestMoveToTracksTheDiscoveredBodyFrame:
    """Both documented body routes reach the target and report the frame used."""

    def test_the_body_hint_route_reaches_and_reports_its_frame(self, hand_body_sim):
        payload = _reach(hand_body_sim)
        assert payload["reached"] is True
        assert payload["position_error_m"] <= _TOL
        assert payload["frame"] == "arm/hand"
        assert payload["frame_type"] == "body"

    def test_the_chain_tail_route_reaches_and_reports_its_frame(self, leaf_body_sim):
        payload = _reach(leaf_body_sim)
        assert payload["reached"] is True
        assert payload["position_error_m"] <= _TOL
        assert payload["frame"] == "arm/link4"
        assert payload["frame_type"] == "body"

    def test_both_routes_resolve_one_physical_frame(self, hand_body_sim, leaf_body_sim):
        """Same kinematics, same tool-mount body, two names - so one answer.

        The two arms differ only in the tip body's name, which is what selects
        the discovery route. A route that resolved a different body would report
        a different pose for an identical scene.
        """
        hand, leaf = _reach(hand_body_sim), _reach(leaf_body_sim)
        assert hand["frame"] != leaf["frame"], "the two arms must exercise different names"
        assert hand["ee_position"] == pytest.approx(leaf["ee_position"])
        assert hand["position_error_m"] == pytest.approx(leaf["position_error_m"])


class TestTheReportedPoseIsTheBodyFrameNotTheInertialFrame:
    """MuJoCo carries two frames per body; the reported one is the frame origin."""

    def test_the_arm_distinguishes_the_two_frames(self, hand_body_sim):
        """Premise: without an inertial offset the next two tests are vacuous."""
        _reach(hand_body_sim)
        data = hand_body_sim._world._data
        bid = _body_id(hand_body_sim, "arm/hand")
        offset = float(np.linalg.norm(np.asarray(data.xipos[bid]) - np.asarray(data.xpos[bid])))
        assert offset > 0.02, f"tip inertial frame only {offset:.4f} m from its origin"

    def test_reported_position_is_the_frame_origin(self, hand_body_sim):
        payload = _reach(hand_body_sim)
        data = hand_body_sim._world._data
        bid = _body_id(hand_body_sim, "arm/hand")
        assert payload["ee_position"] == pytest.approx(list(np.asarray(data.xpos[bid], dtype=float)))
        assert payload["ee_position"] != pytest.approx(list(np.asarray(data.xipos[bid], dtype=float)))

    def test_reported_orientation_is_the_frame_quaternion(self, hand_body_sim):
        payload = _reach(hand_body_sim)
        data = hand_body_sim._world._data
        bid = _body_id(hand_body_sim, "arm/hand")
        assert payload["ee_orientation_wxyz"] == pytest.approx(list(np.asarray(data.xquat[bid], dtype=float)))


class TestTheSolverAndTheConvergenceCheckShareOneFrame:
    """The frame the IK optimizes is the frame ``reached`` is decided at."""

    def test_position_error_is_measured_at_the_reported_frame(self, hand_body_sim):
        payload = _reach(hand_body_sim)
        expected = float(np.linalg.norm(np.asarray(payload["ee_position"]) - np.asarray(REACHABLE, dtype=float)))
        assert payload["position_error_m"] == pytest.approx(expected)

    def test_the_ik_bridge_reads_the_same_frame_back(self, hand_body_sim):
        """mink's forward pose for the settled configuration is the reported one.

        A readback of a different frame of the same body would leave the solver
        optimizing one point and the servo measuring another.
        """
        from strands_robots.simulation.ik import MinkIKBridge

        payload = _reach(hand_body_sim)
        model, data = hand_body_sim._world._model, hand_body_sim._world._data
        bridge = MinkIKBridge(model, payload["frame"], payload["frame_type"])
        pose = bridge.ee_pose(np.array(data.qpos, dtype=np.float64, copy=True))
        assert list(pose[:3, 3]) == pytest.approx(payload["ee_position"])


def _body_id(sim: Any, name: str) -> int:
    """Compiled body index for ``name``."""
    import mujoco as mj

    bid = int(mj.mj_name2id(sim._world._model, mj.mjtObj.mjOBJ_BODY, name))
    assert bid >= 0, f"body {name!r} not in the compiled model"
    return bid
