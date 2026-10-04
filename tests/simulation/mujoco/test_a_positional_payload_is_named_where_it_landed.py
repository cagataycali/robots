"""A payload passed positionally to a robot_name-first primitive is named in the refusal.

``set_gripper("close")`` binds "close" to ``robot_name``; the refusal used to read
``'state' must be "open" or "close", got None`` - a value the caller never typed.
"""

import pytest

pytest.importorskip("mujoco")

from strands_robots import Robot  # noqa: E402


@pytest.fixture(scope="module")
def so101():
    robot = Robot("so101", mesh=False)
    yield robot
    robot.destroy()


@pytest.mark.parametrize(
    ("verb", "payload", "param"),
    [("set_gripper", "close", "state"), ("move_to", [0.2, 0.0, 0.1], "position"), ("rotate_wrist", 0.3, "target_yaw")],
)
def test_the_refusal_names_robot_name_as_the_slot_that_received_it(so101, verb, payload, param) -> None:
    result = getattr(so101, verb)(payload)
    text = result["content"][0]["text"]
    assert result["status"] == "error"
    assert f"'robot_name', and it received {payload!r}; name the argument instead: {verb}({param}=...)" in text
    assert "got None" not in text
    assert getattr(so101, verb)(**{param: payload})["status"] == "success"
