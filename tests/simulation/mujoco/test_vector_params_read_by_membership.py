"""Regression tests: caller-supplied vectors are read by membership, not truthiness.

Every pose/wrench parameter in the MuJoCo scene API is optional, and the engine
has to decide whether the caller supplied one. Deciding that by testing the
vector itself (``if position:``, ``position or <default>``) is wrong twice over:

* A NumPy array has no boolean value, so ``position or <default>`` raises a bare
  ``ValueError: The truth value of an array with more than one element is
  ambiguous`` straight through the ``{"status": ...}`` tool-result contract - and
  an array is exactly what pose arithmetic (``base + offset``), an observation
  row or a computed wrench (``mass * accel``) produces.
* An empty vector is falsy, so it reads as "omitted": the default is substituted
  (``add_camera`` placed the camera at ``[1, 1, 1]``) or the write is skipped
  (``move_object`` moved nothing), and the call still reports success. On a
  static object the empty vector reached MuJoCo's spec setter and raised a bare
  pybind ``TypeError``.

The contract these pin: a vector parameter is supplied when it is not ``None``;
a supplied vector is validated for length/finiteness and normalized to plain
floats; and the status text reports what was actually applied.
"""

import numpy as np
import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


def _text(result):
    return " ".join(block["text"] for block in result["content"] if "text" in block)


def _camera_position(sim, camera_name):
    """Return the camera's world position from its reported extrinsics."""
    params = sim.get_camera_params(camera_name=camera_name)
    return [float(v) for v in params.T_world_cam[:3, 3]]


def _position_of(sim, body):
    result = sim.get_body_state(body)
    assert result["status"] == "success", result
    payload = [block["json"] for block in result["content"] if "json" in block][0]
    return payload["position"]


@pytest.fixture
def sim():
    s = Simulation(tool_name="test_vector_membership_sim", mesh=False)
    s.create_world()
    yield s
    s.cleanup()


class TestNumpyVectorsAreAccepted:
    """A NumPy vector is honored, not raised past the tool-result contract."""

    def test_add_object_accepts_numpy_pose_and_normalizes_it(self, sim):
        """``add_object`` places the object at the NumPy pose it was given.

        The status text and the stored object echo plain floats: a NumPy element
        surviving this boundary renders as ``np.float64(0.05)`` in agent-visible
        output and contradicts ``SimObject``'s ``list[float]`` annotation.
        """
        base = np.array([0.2, 0.1, 0.05])
        result = sim.add_object(
            name="cube",
            shape="box",
            size=np.array([0.05, 0.05, 0.05]),
            position=base + np.array([0.1, 0.0, 0.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        assert result["status"] == "success", result
        assert _position_of(sim, "cube") == pytest.approx([0.3, 0.1, 0.05], abs=1e-6)
        assert "np.float64" not in _text(result)
        assert "np.float64" not in _text(sim.list_objects())

    def test_add_camera_accepts_numpy_pose_and_looks_where_asked(self, sim):
        """``add_camera`` honors a NumPy position/target instead of raising.

        Pinned on the rendered camera's reported pose, not on the call status:
        the pre-fix ``position or <default>`` failure mode for a wrong-typed
        vector was a silent fallback to ``[1, 1, 1]``.
        """
        result = sim.add_camera(
            name="look",
            position=np.array([0.9, 0.0, 0.55]),
            target=np.array([0.3, 0.0, 0.1]),
        )
        assert result["status"] == "success", result
        assert _camera_position(sim, "look") == pytest.approx([0.9, 0.0, 0.55], abs=1e-6)

    def test_add_robot_accepts_numpy_position(self, sim):
        """``add_robot`` honors a NumPy base position instead of raising."""
        result = sim.add_robot(name="panda", position=np.array([0.0, 0.15, 0.0]))
        assert result["status"] == "success", result
        assert sim._world.robots["panda"].position == pytest.approx([0.0, 0.15, 0.0], abs=1e-6)

    def test_move_object_accepts_numpy_pose(self, sim):
        """``move_object`` moves the body to the NumPy pose it was given."""
        sim.add_object(name="cube", shape="box", size=[0.05, 0.05, 0.05], position=[0.4, 0.0, 0.05])
        result = sim.move_object(name="cube", position=np.array([0.1, 0.2, 0.3]))
        assert result["status"] == "success", result
        assert _position_of(sim, "cube") == pytest.approx([0.1, 0.2, 0.3], abs=1e-6)
        assert "np.float64" not in _text(result)

    def test_move_object_accepts_numpy_pose_on_static_object(self, sim):
        """The static (spec-recompile) path honors a NumPy pose too."""
        sim.add_object(name="fixture", shape="box", size=[0.05, 0.05, 0.05], position=[-0.4, 0.0, 0.05], is_static=True)
        result = sim.move_object(name="fixture", position=np.array([-0.2, 0.1, 0.05]))
        assert result["status"] == "success", result
        assert _position_of(sim, "fixture") == pytest.approx([-0.2, 0.1, 0.05], abs=1e-6)

    def test_apply_force_accepts_numpy_wrench(self, sim):
        """``apply_force`` accepts a NumPy force/torque/point and moves the body.

        A computed wrench is an array; assert the body actually accelerated
        along it so the force reached ``qfrc_applied`` rather than being
        reported as applied.
        """
        sim.add_object(name="puck", shape="box", size=[0.05, 0.05, 0.05], position=[0.0, 0.0, 0.05])
        result = sim.apply_force(
            body_name="puck",
            force=np.array([8.0, 0.0, 0.0]),
            torque=np.array([0.0, 0.0, 0.0]),
            point=np.array([0.0, 0.0, 0.05]),
        )
        assert result["status"] == "success", result
        sim.step(50)
        assert _position_of(sim, "puck")[0] > 0.02, "NumPy force should have pushed the puck along +x"


class TestEmptyVectorIsRejectedNotTreatedAsOmitted:
    """An empty vector is a wrong-length request, never an omission."""

    def test_add_camera_rejects_empty_position(self, sim):
        """An empty position errors instead of placing the camera at the default."""
        result = sim.add_camera(name="cam", position=[], target=[0.0, 0.0, 0.1])
        assert result["status"] == "error", result
        assert "'position' must be a 3-element vector" in _text(result)
        assert "cam" not in sim.list_cameras()

    def test_add_camera_rejects_empty_target(self, sim):
        """An empty target errors instead of aiming at the default origin."""
        result = sim.add_camera(name="cam", position=[1.0, 0.0, 0.5], target=[])
        assert result["status"] == "error", result
        assert "'target' must be a 3-element vector" in _text(result)

    def test_move_object_rejects_empty_position(self, sim):
        """An empty position errors and the object stays where it was."""
        sim.add_object(name="cube", shape="box", size=[0.05, 0.05, 0.05], position=[0.4, 0.0, 0.05])
        result = sim.move_object(name="cube", position=[])
        assert result["status"] == "error", result
        assert "'position' must be a 3-element vector" in _text(result)
        assert _position_of(sim, "cube") == pytest.approx([0.4, 0.0, 0.05], abs=1e-6)

    def test_move_object_rejects_empty_orientation(self, sim):
        """An empty orientation errors rather than reporting a move."""
        sim.add_object(name="cube", shape="box", size=[0.05, 0.05, 0.05], position=[0.4, 0.0, 0.05])
        result = sim.move_object(name="cube", orientation=())
        assert result["status"] == "error", result
        assert "'orientation' must be a 4-element vector" in _text(result)

    def test_move_object_rejects_empty_position_on_static_object(self, sim):
        """The static path reports a structured error, not a bare pybind TypeError."""
        sim.add_object(name="fixture", shape="box", size=[0.05, 0.05, 0.05], position=[-0.4, 0.0, 0.05], is_static=True)
        result = sim.move_object(name="fixture", position=[])
        assert result["status"] == "error", result
        assert "'position' must be a 3-element vector" in _text(result)
        assert _position_of(sim, "fixture") == pytest.approx([-0.4, 0.0, 0.05], abs=1e-6)

    def test_omitted_vectors_still_take_their_documented_defaults(self, sim):
        """Omission (``None``) keeps working: the documented defaults apply."""
        assert sim.add_object(name="cube", shape="box", size=[0.05, 0.05, 0.05])["status"] == "success"
        assert _position_of(sim, "cube") == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
        assert sim.add_camera(name="cam")["status"] == "success"
        assert _camera_position(sim, "cam") == pytest.approx([1.0, 1.0, 1.0], abs=1e-6)


class TestMoveObjectReportsWhatItApplied:
    """The success text names the components actually written."""

    def test_orientation_only_move_is_not_reported_as_unchanged(self, sim):
        """An orientation-only move names the orientation, not ``same``.

        The rotation IS applied on this path, so reporting "moved to same"
        (what ``position or 'same'`` produced) contradicts the state.
        """
        sim.add_object(name="cube", shape="box", size=[0.05, 0.05, 0.05], position=[0.4, 0.0, 0.05])
        result = sim.move_object(name="cube", orientation=[0.7071, 0.0, 0.0, 0.7071])
        assert result["status"] == "success", result
        text = _text(result)
        assert "orientation" in text
        assert "same" not in text

    def test_both_components_are_reported(self, sim):
        """A full pose move names both components."""
        sim.add_object(name="cube", shape="box", size=[0.05, 0.05, 0.05], position=[0.4, 0.0, 0.05])
        result = sim.move_object(name="cube", position=[0.1, 0.0, 0.2], orientation=[1.0, 0.0, 0.0, 0.0])
        assert result["status"] == "success", result
        text = _text(result)
        assert "position" in text and "orientation" in text

    def test_no_components_reports_unchanged(self, sim):
        """Omitting both components is a documented no-op reported as ``same``."""
        sim.add_object(name="cube", shape="box", size=[0.05, 0.05, 0.05], position=[0.4, 0.0, 0.05])
        result = sim.move_object(name="cube")
        assert result["status"] == "success", result
        assert "same" in _text(result)
