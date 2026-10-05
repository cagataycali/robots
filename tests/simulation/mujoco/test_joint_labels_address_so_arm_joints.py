"""SO-arm joints can be addressed by what they do, not only by servo id.

Observed: an agent asked to lift the SO-101's shoulder sent
``set_joint_positions {"Shoulder_Lift": 0.3}`` and was refused - the asset
names its joints ``1``..``6`` and nothing in the sim mapped them to the
``shoulder_pan .. gripper`` the same arm's driver and datasets use. The agent
exported the 14 KB MJCF to guess. The registry now carries ``joint_labels``
for the SO arms; ``get_robot_state`` prints ``1 (shoulder_pan)`` and the joint
write paths and ``send_action`` accept the label (bare, ``<robot>/<label>``,
any case).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("mujoco")

from strands_robots.registry import joint_labels  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

SO_LABELS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def _asset_file(pack: str, filename: str) -> str:
    """Path to a file in an already-downloaded asset pack, or skip.

    Gated on the pack *directory* so a host without the pack skips rather than
    reaching for the network, and the declared file is then asserted.
    """
    from strands_robots.utils import get_search_paths

    for root in get_search_paths():
        if (Path(root) / pack).is_dir():
            path = Path(root) / pack / filename
            assert path.is_file(), f"{pack} is present but does not carry {filename}"
            return str(path)
    pytest.skip(f"asset pack '{pack}' is not available")


def _text(result: dict) -> str:
    return result["content"][0]["text"]


def _json(result: dict) -> dict:
    return result["content"][1]["json"]


@pytest.fixture
def so101():
    sim = Simulation()
    sim.create_world()
    res = sim.add_robot(name="so101", data_config="so101")
    if res["status"] != "success":
        pytest.skip(f"so101 not available: {_text(res)}")
    sim.reset()
    yield sim
    sim.destroy()


class TestRegistry:
    def test_so101_labels_follow_the_servo_ids(self):
        assert joint_labels("so101") == dict(zip([str(i) for i in range(1, 7)], SO_LABELS, strict=True))

    @pytest.mark.parametrize("robot", ["so100", "lekiwi"])
    def test_every_so_arm100_arm_labels_its_cad_names_the_same(self, robot):
        """LeKiwi carries the SO-100 arm, so the arm's labels are the SO-100's."""
        arm = {jnt: lbl for jnt, lbl in joint_labels(robot).items() if lbl in SO_LABELS}
        assert arm == joint_labels("so100")
        assert "Rotation" in arm and "Jaw" in arm

    def test_an_alias_resolves_too(self):
        assert joint_labels("so101_follower") == joint_labels("so101")

    def test_no_labels_is_an_empty_dict(self):
        assert joint_labels("panda") == {}
        assert joint_labels("no-such-robot") == {}


def test_lekiwi_takes_the_so_arm_labels_in_send_action():
    """The dict an SO-100 accepts drives LeKiwi's arm too, wheels by actuator name."""
    _asset_file("lekiwi", "lekiwi/lekiwi.xml")
    sim = Simulation()
    sim.create_world()
    try:
        assert sim.add_robot(name="lekiwi", data_config="lekiwi")["status"] == "success"
        action = dict.fromkeys(SO_LABELS, 0.1) | dict.fromkeys(
            ["base_back_wheel", "base_left_wheel", "base_right_wheel"], 0.0
        )
        result = sim.send_action(action, robot_name="lekiwi")
        assert result["status"] == "success", _text(result)
    finally:
        sim.destroy()


class TestGetRobotState:
    def test_text_shows_the_label_beside_the_joint(self, so101):
        text = _text(so101._dispatch_action("get_robot_state", {"robot_name": "so101"}))
        assert "1 (shoulder_pan): pos=" in text
        assert "6 (gripper): pos=" in text

    def test_json_keeps_raw_keys_and_adds_the_map(self, so101):
        payload = _json(so101._dispatch_action("get_robot_state", {}))
        assert set(payload["state"]) == {str(i) for i in range(1, 7)}
        assert payload["joint_labels"] == joint_labels("so101")


class TestJointWrites:
    def test_a_bare_label_writes_the_joint(self, so101):
        result = so101._dispatch_action("set_joint_positions", {"positions": {"shoulder_lift": 0.3}})
        assert result["status"] == "success", _text(result)
        state = _json(so101._dispatch_action("get_robot_state", {}))["state"]
        assert state["2"]["position"] == pytest.approx(0.3)

    def test_label_case_is_not_information(self, so101):
        result = so101._dispatch_action("set_joint_positions", {"positions": {"Shoulder_Lift": 0.3}})
        assert result["status"] == "success", _text(result)
        assert _json(so101._dispatch_action("get_robot_state", {}))["state"]["2"]["position"] == pytest.approx(0.3)

    def test_qualified_label_and_raw_name_mix_in_one_write(self, so101):
        result = so101._dispatch_action(
            "set_joint_positions", {"positions": {"so101/wrist_roll": 0.2, "1": 0.1}, "robot_name": "so101"}
        )
        assert result["status"] == "success", _text(result)
        state = _json(so101._dispatch_action("get_robot_state", {}))["state"]
        assert state["5"]["position"] == pytest.approx(0.2)
        assert state["1"]["position"] == pytest.approx(0.1)

    def test_hold_moves_the_servo_setpoint_through_the_label(self, so101):
        result = so101._dispatch_action("set_joint_positions", {"positions": {"shoulder_lift": 0.3}, "hold": True})
        assert result["status"] == "success", _text(result)
        assert "setpoint(s) moved with it" in _text(result)

    def test_velocities_take_the_label_too(self, so101):
        result = so101._dispatch_action("set_joint_velocities", {"velocities": {"gripper": 0.1}})
        assert result["status"] == "success", _text(result)

    def test_an_unknown_key_is_still_refused_and_the_error_teaches_the_labels(self, so101):
        result = so101._dispatch_action("set_joint_positions", {"positions": {"elbow": 0.3}})
        assert result["status"] == "error"
        text = _text(result)
        assert "Joint 'elbow' not found" in text
        assert "may also be written by label" in text
        assert "3=elbow_flex" in text

    def test_a_label_scoped_to_the_wrong_robot_is_refused(self, so101):
        result = so101._dispatch_action("set_joint_positions", {"positions": {"nobody/shoulder_lift": 0.3}})
        assert result["status"] == "error"

    def test_a_robot_without_labels_is_unchanged(self):
        sim = Simulation()
        sim.create_world()
        res = sim.add_robot(name="panda", data_config="panda")
        if res["status"] != "success":
            sim.destroy()
            pytest.skip(_text(res))
        try:
            result = sim._dispatch_action("set_joint_positions", {"positions": {"shoulder_lift": 0.3}})
            assert result["status"] == "error"
            assert "may also be written by label" not in _text(result)
            assert "(" not in _text(sim._dispatch_action("get_robot_state", {})).split("\n")[1]
        finally:
            sim.destroy()


class TestSendAction:
    """``send_action`` takes the label ``set_joint_positions`` takes, for the same joint."""

    @pytest.mark.parametrize("key", ["shoulder_lift", "Shoulder_Lift", "so101/shoulder_lift"])
    def test_a_label_writes_the_same_ctrl_as_the_servo_id(self, so101, key):
        data = so101._world._data
        assert so101.send_action({"2": 0.3}, robot_name="so101")["status"] == "success"
        by_id = data.ctrl.copy()
        data.ctrl[:] = 0.0
        result = so101.send_action({key: 0.3}, robot_name="so101")
        assert result["status"] == "success", _text(result)
        assert list(data.ctrl) == list(by_id)

    def test_an_unknown_key_is_refused_and_the_error_teaches_the_labels(self, so101):
        result = so101.send_action({"elbow": 0.3}, robot_name="so101")
        assert result["status"] == "error"
        assert _json(result)["unresolved_keys"] == ["elbow"]
        assert "may also be written by label" in _text(result)
        assert "3=elbow_flex" in _text(result)


class TestTheShortFormIsLabelledToo:
    """``add_robot("so101")`` - what the quickstart teaches - carries the labels.

    ``add_robot`` resolves the model from ``data_config`` when given and from
    the instance ``name`` otherwise, so a robot added by the short form comes
    from a registry entry just as much as one naming ``data_config``.
    """

    @pytest.fixture
    def so101_short_form(self):
        sim = Simulation()
        sim.create_world()
        res = sim.add_robot(name="so101")  # no data_config
        if res["status"] != "success":
            sim.destroy()
            pytest.skip(f"so101 not available: {_text(res)}")
        sim.reset()
        yield sim
        sim.destroy()

    def test_the_state_carries_the_labels(self, so101_short_form):
        result = so101_short_form._dispatch_action("get_robot_state", {})
        assert "2 (shoulder_lift): pos=" in _text(result)
        assert _json(result)["joint_labels"] == joint_labels("so101")

    def test_a_label_writes_the_joint(self, so101_short_form):
        result = so101_short_form._dispatch_action("set_joint_positions", {"positions": {"shoulder_lift": 0.3}})
        assert result["status"] == "success", _text(result)
        state = _json(so101_short_form._dispatch_action("get_robot_state", {}))["state"]
        assert state["2"]["position"] == pytest.approx(0.3)

    def test_a_distinct_instance_label_reads_the_data_config(self):
        """The form ``add_robot``'s own deprecation hint recommends.

        ``add_robot(name="arm0", data_config="so101")`` labels the SO-101 it
        loaded, not the nothing that ``arm0`` names in the registry.
        """
        sim = Simulation()
        sim.create_world()
        res = sim.add_robot(name="arm0", data_config="so101")
        if res["status"] != "success":
            sim.destroy()
            pytest.skip(_text(res))
        try:
            result = sim._dispatch_action("set_joint_positions", {"positions": {"arm0/shoulder_lift": 0.3}})
            assert result["status"] == "success", _text(result)
            state = sim._dispatch_action("get_robot_state", {})
            assert _json(state)["state"]["2"]["position"] == pytest.approx(0.3)
            assert "2 (shoulder_lift): pos=" in _text(state)
        finally:
            sim.destroy()

    def test_a_colliding_instance_label_cannot_mislabel_a_foreign_model(self):
        """An instance label that happens to name another registry entry.

        The labels are keyed by the asset's joint name, so the mismatch can
        only fail to match - never move the joint the label does not name.
        """
        so101_xml = _asset_file("robotstudio_so101", "so101_new_calib.xml")
        sim = Simulation()
        sim.create_world()
        res = sim.add_robot(name="so100", urdf_path=so101_xml)
        if res["status"] != "success":
            sim.destroy()
            pytest.skip(_text(res))
        try:
            state = sim._dispatch_action("get_robot_state", {})
            assert "joint_labels" not in _json(state)
            assert "(" not in _text(state).split("\n")[1]
            result = sim._dispatch_action("set_joint_positions", {"positions": {"shoulder_lift": 0.3}})
            assert result["status"] == "error"
            assert "may also be written by label" not in _text(result)
        finally:
            sim.destroy()


@pytest.mark.parametrize("robot", ["lekiwi", "so100", "so101"])
def test_the_robot_page_table_names_what_each_side_of_the_api_speaks(robot):
    """The page's left column is what ``get_observation`` returns; its right column is what ``send_action`` also takes.

    Observed: a reader of the SO-101 page wrote ``send_action({"shoulder_pan": v})``
    (accepted), then read ``obs["shoulder_pan"]`` back and hit ``KeyError`` - the
    observation keeps the model's joint name, and the table's two columns were
    headed "Model joint" / "Action key", which never said which one a read returns.
    The fence above the table prints ``robot_action_keys`` (actuator names), so
    the header says the labels are accepted as well as those, not instead of them.
    """
    from tests._docs_hooks import docs_hook

    page = docs_hook("robot_pages").robot_page(robot)
    header = "| Observation key | also accepted by `send_action` |"
    assert header in page, f"the {robot} page does not name the observation side of its joint table"
    rows = page.split(header, 1)[1].split("\n\n", 1)[0].splitlines()[2:]
    table = dict(re.findall(r"\| `([^`]+)` \| `([^`]+)` \|", "\n".join(rows)))
    assert table == joint_labels(robot)

    sim = Simulation()
    sim.create_world()
    res = sim.add_robot(name=robot)
    if res["status"] != "success":
        sim.destroy()
        pytest.skip(_text(res))
    try:
        for key in sim.robot_action_keys(robot):
            assert sim.send_action({key: 0.0}, robot_name=robot)["status"] == "success", key
        for obs_key, label in table.items():
            result = sim.send_action({label: 0.1}, robot_name=robot)
            assert result["status"] == "success", _text(result)
            obs = sim.get_observation(robot_name=robot, skip_images=True)
            assert obs_key in obs and label not in obs
    finally:
        sim.destroy()
