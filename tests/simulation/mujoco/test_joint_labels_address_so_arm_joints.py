"""SO-arm joints can be addressed by what they do, not only by servo id.

Observed: an agent asked to lift the SO-101's shoulder sent
``set_joint_positions {"Shoulder_Lift": 0.3}`` and was refused - the asset
names its joints ``1``..``6`` and nothing in the sim mapped them to the
``shoulder_pan .. gripper`` the same arm's driver and datasets use. The agent
exported the 14 KB MJCF to guess. The registry now carries ``joint_labels``
for the SO arms; ``get_robot_state`` prints ``1 (shoulder_pan)`` and the joint
write paths accept the label (bare, ``<robot>/<label>``, any case).
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.registry import joint_labels  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

SO_LABELS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


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

    def test_so100_labels_cover_the_cad_names(self):
        labels = joint_labels("so100")
        assert list(labels.values()) == SO_LABELS
        assert "Rotation" in labels and "Jaw" in labels

    def test_an_alias_resolves_too(self):
        assert joint_labels("so101_follower") == joint_labels("so101")

    def test_no_labels_is_an_empty_dict(self):
        assert joint_labels("panda") == {}
        assert joint_labels("no-such-robot") == {}


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
