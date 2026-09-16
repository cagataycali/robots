"""``get_features`` counts the list it is introducing, not the whole world.

The report is a label-and-list pair per line -- ``Joints (N): a, b, c`` -- and
the same pairing in the json (``n_joints`` beside ``joint_names``). The counts
were read from the compiled model (``model.njnt`` / ``model.nu`` /
``model.ncam``) while the lists were scoped to ``robot_name``, so a robot-scoped
listing labelled a robot's own parts with the whole scene's totals: a 6-joint
arm in a 3-robot world read ``Joints (42): 1, 2, 3, 4, 5, 6``. The right numbers
were already in the same payload two lines down, where the ``robots`` map
reports ``arm0: 6 joints, 6 actuators``.

Cameras were worse than mislabelled. Scoping filtered camera NAMES by the
robot's namespace prefix, but a camera is mounted rather than named: the one
``add_camera(parent_body="g1/pelvis")`` puts on a robot's pelvis carries no
namespace in its own name, so the filter dropped every camera a robot wears and
the line read ``Cameras (2): none (free camera only)`` for a G1 with a camera
riding on it. That is the same correction the actuator line one above already
needed (ownership, not prefix spelling -- see
``test_action_keys_follow_resolved_actuator_ownership``), and membership of the
mount body is read the way :meth:`_body_is_namespaced` reads it, from the
compiled name.

``create_world`` always compiles a ``default`` camera, so the "free camera only"
note was unreachable for a whole-world listing: the only listing that ever
printed it was a robot-scoped one, where it was always false. A scene loaded
from an MJCF that declares no camera is the case it is true for, and it is
pinned below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation

# One hinge, one position servo, one declared camera. Hand-written so the spec
# compiles with nothing downloaded.
_ROBOT_XML = """
<mujoco model="{model}">
  <compiler angle="radian"/>
  <worldbody>
    <body name="link" pos="0 0 0.2">
      <joint name="pan" type="hinge" axis="0 0 1" range="-2 2" damping="1"/>
      <geom name="body_geom" type="box" size="0.05 0.05 0.05"/>
      <camera name="{camera}" pos="0.3 0 0.1" xyaxes="0 -1 0 0 0 1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="pan_act" joint="pan" kp="10"/>
  </actuator>
</mujoco>
"""

# A robot whose camera is declared outside every one of its bodies, which is how
# the compiled camera ends up on the world body while still being the robot's.
_ROBOT_XML_LOOSE_CAMERA = """
<mujoco model="{model}">
  <compiler angle="radian"/>
  <worldbody>
    <camera name="{camera}" pos="1.5 0 1" xyaxes="0 1 0 -0.5 0 1"/>
    <body name="link" pos="0 0 0.2">
      <joint name="pan" type="hinge" axis="0 0 1" range="-2 2" damping="1"/>
      <geom name="body_geom" type="box" size="0.05 0.05 0.05"/>
    </body>
  </worldbody>
  <actuator>
    <position name="pan_act" joint="pan" kp="10"/>
  </actuator>
</mujoco>
"""

# The same robot without the declared camera, for a robot that wears none.
_ROBOT_XML_NO_CAMERA = """
<mujoco model="{model}">
  <compiler angle="radian"/>
  <worldbody>
    <body name="link" pos="0 0 0.2">
      <joint name="pan" type="hinge" axis="0 0 1" range="-2 2" damping="1"/>
      <geom name="body_geom" type="box" size="0.05 0.05 0.05"/>
    </body>
  </worldbody>
  <actuator>
    <position name="pan_act" joint="pan" kp="10"/>
  </actuator>
</mujoco>
"""

# A scene with no camera at all, which is what the free-camera note is about.
_BARE_SCENE_XML = """
<mujoco model="bare">
  <compiler angle="radian"/>
  <worldbody>
    <light pos="0 0 3"/>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="cube" pos="0 0 0.3">
      <freejoint name="cube_free"/>
      <geom name="cube_geom" type="box" size="0.05 0.05 0.05"/>
    </body>
  </worldbody>
</mujoco>
"""


def _write_robot(tmp_path: Path, model: str, camera: str) -> str:
    path = tmp_path / f"{model}.xml"
    path.write_text(_ROBOT_XML.format(model=model, camera=camera))
    return str(path)


def _text(sim: Any, robot_name: str | None = None) -> str:
    return str(sim.get_features(robot_name=robot_name)["content"][0]["text"])


def _features(sim: Any, robot_name: str | None = None) -> dict[str, Any]:
    payload = sim.get_features(robot_name=robot_name)["content"][1]["json"]["features"]
    return dict(payload)


def _line(text: str, label: str) -> str:
    return next(ln for ln in text.splitlines() if ln.startswith(f"{label} ("))


def _counted(text: str, label: str) -> tuple[int, list[str]]:
    """The count a line declares and the names it then lists."""
    line = _line(text, label)
    count = int(line.split("(", 1)[1].split(")", 1)[0])
    listed = line.split("): ", 1)[1]
    names = [] if listed.startswith("none") else [n for n in (p.strip() for p in listed.split(",")) if n]
    return count, names


@pytest.fixture
def sim():
    s = Simulation(tool_name="test_features_counts_label_their_lists", mesh=False)
    s.create_world(gravity=[0, 0, -9.81])
    yield s
    s.cleanup(policy_stop_timeout=0.5)


@pytest.fixture
def two_robots(sim, tmp_path):
    """``rover`` (declares camera ``front``) then ``arm`` (declares ``wrist``)."""
    assert sim.add_robot("rover", urdf_path=_write_robot(tmp_path, "rover_model", "front"))["status"] == "success"
    assert sim.add_robot("arm", urdf_path=_write_robot(tmp_path, "arm_model", "wrist"))["status"] == "success"
    return sim


class TestACountCountsTheListItIntroduces:
    @pytest.mark.parametrize("label", ["Joints", "Actuators", "Cameras"])
    def test_a_robot_scoped_line_counts_only_that_robots_parts(self, two_robots, label) -> None:
        """Each line's count is the length of the list printed after it."""
        for robot in ("rover", "arm"):
            count, names = _counted(_text(two_robots, robot), label)
            assert count == len(names) == 1, f"{label} for {robot}: {count} labelling {names}"

    def test_the_json_counts_match_the_name_lists_beside_them(self, two_robots) -> None:
        """``n_joints`` / ``n_actuators`` / ``n_cameras`` count their own lists."""
        for scope in (None, "rover", "arm"):
            payload = _features(two_robots, scope)
            for count_key, names_key in (
                ("n_joints", "joint_names"),
                ("n_actuators", "actuator_names"),
                ("n_cameras", "camera_names"),
            ):
                assert payload[count_key] == len(payload[names_key]), f"{count_key} in scope {scope!r}"

    def test_the_scoped_counts_agree_with_the_robots_map_below_them(self, two_robots) -> None:
        """The header reports what the per-robot entry in the same payload does."""
        payload = _features(two_robots, "rover")
        entry = payload["robots"]["rover"]
        assert (payload["n_joints"], payload["n_actuators"]) == (entry["n_joints"], entry["n_actuators"])

    def test_the_whole_world_listing_still_reports_the_whole_world(self, two_robots) -> None:
        """Both robots' parts and both cameras, counted."""
        for label, expected in (("Joints", 2), ("Actuators", 2), ("Cameras", 3)):
            count, names = _counted(_text(two_robots), label)
            assert count == len(names) == expected, f"{label}: {count} labelling {names}"


class TestACameraBelongsToTheRobotItIsMountedOn:
    def test_a_camera_a_robot_declares_is_listed_for_it_and_no_other(self, two_robots) -> None:
        assert _counted(_text(two_robots, "rover"), "Cameras")[1] == ["rover/front"]
        assert _counted(_text(two_robots, "arm"), "Cameras")[1] == ["arm/wrist"]

    def test_a_camera_mounted_on_the_robot_later_is_the_robots_own(self, two_robots) -> None:
        """``add_camera(parent_body=...)`` names no namespace, but it rides along."""
        added = two_robots.add_camera(name="chase", parent_body="rover/link", position=[-1, 0, 0.5])
        assert added["status"] == "success", added
        assert _counted(_text(two_robots, "rover"), "Cameras") == (2, ["rover/front", "chase"])
        assert _counted(_text(two_robots, "arm"), "Cameras") == (1, ["arm/wrist"])

    def test_a_camera_the_robot_brought_is_its_own_wherever_it_is_declared(self, sim, tmp_path) -> None:
        """Declared outside the robot's bodies, so only the registry knows it."""
        for robot in ("one", "two"):
            path = tmp_path / f"{robot}.xml"
            path.write_text(_ROBOT_XML_LOOSE_CAMERA.format(model=f"{robot}_model", camera="wrist"))
            assert sim.add_robot(robot, urdf_path=str(path))["status"] == "success"
        assert _counted(_text(sim, "one"), "Cameras")[1] == ["one/wrist"]
        assert _counted(_text(sim, "two"), "Cameras")[1] == ["two/wrist"]

    def test_a_robot_wearing_none_does_not_call_the_scene_free_camera_only(self, sim, tmp_path) -> None:
        """The scene's cameras are still there; this robot just has none."""
        path = tmp_path / "plain.xml"
        path.write_text(_ROBOT_XML_NO_CAMERA.format(model="plain"))
        assert sim.add_robot("plain", urdf_path=str(path))["status"] == "success"
        assert "default" in sim.list_cameras(), "the scene does have a camera to render from"
        line = _line(_text(sim, "plain"), "Cameras")
        assert line.startswith("Cameras (0): none mounted on 'plain'")
        assert "free camera only" not in line
        assert "list_cameras" in line

    def test_a_scene_with_no_camera_at_all_is_the_free_cameras(self, sim, tmp_path) -> None:
        """The one case the free-camera note is true for."""
        path = tmp_path / "bare.xml"
        path.write_text(_BARE_SCENE_XML)
        assert sim.load_scene(str(path))["status"] == "success"
        assert _line(_text(sim), "Cameras") == "Cameras (0): none (free camera only)"
