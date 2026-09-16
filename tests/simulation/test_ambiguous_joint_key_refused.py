"""A bare joint key two robots carry is refused by the dict-form joint writes.

``_resolve_mj_name``'s first-match fallback is documented as an "unambiguous
or explicit" contract, but ``set_joint_positions`` / ``set_joint_velocities``
in dict form did not enforce it: in a scene with two so101s,
``positions={"1": 0.3}`` moved the first robot attached and reported success,
while the list form, ``get_robot_state``, ``move_to`` and ``run_policy`` on the
same scene all refused to guess. Now the dict form refuses too, naming the
robots that carry the key and both remedies.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine


def _text(result: dict) -> str:
    return "\n".join(c["text"] for c in result["content"] if isinstance(c, dict) and "text" in c)


def _q1(sim, robot: str) -> float:
    return float(sim.get_observation(robot, skip_images=True)["1"])


@pytest.fixture
def two_arms():
    sim = MuJoCoSimEngine(tool_name="two_arms", mesh=False)
    sim.create_world()
    assert sim.add_robot(name="so101", data_config="so101")["status"] == "success"
    assert sim.add_robot(name="arm2", data_config="so101", position=[0.5, 0, 0])["status"] == "success"
    try:
        yield sim
    finally:
        sim.cleanup()


@pytest.mark.parametrize("key", ["1", "shoulder_pan", "Shoulder_Pan"])
def test_bare_key_carried_by_both_robots_is_refused_and_writes_nothing(two_arms, key):
    before = _q1(two_arms, "so101")
    r = two_arms.set_joint_positions({key: 0.3})
    assert r["status"] == "error"
    text = _text(r)
    assert text.startswith(
        f"set_joint_positions: joint key '{key}' is ambiguous - robots 'so101' and 'arm2' each carry it, so nothing was written."
    )
    assert f"Pass robot_name= to scope the write, or qualify the key ('so101/{key}' or 'arm2/{key}')." in text
    assert _q1(two_arms, "so101") == before


def test_set_joint_velocities_refuses_the_same_way(two_arms):
    r = two_arms.set_joint_velocities({"1": 0.1})
    assert r["status"] == "error"
    assert _text(r).startswith("set_joint_velocities: joint key '1' is ambiguous")


def test_robot_name_scope_and_qualified_key_still_write(two_arms):
    assert two_arms.set_joint_positions({"1": 0.3}, robot_name="arm2")["status"] == "success"
    assert two_arms.set_joint_positions({"so101/1": 0.2})["status"] == "success"
    assert _q1(two_arms, "arm2") == pytest.approx(0.3)
    assert _q1(two_arms, "so101") == pytest.approx(0.2)


def test_single_robot_bare_key_is_unchanged():
    sim = MuJoCoSimEngine(tool_name="one_arm", mesh=False)
    sim.create_world()
    assert sim.add_robot(name="so101", data_config="so101")["status"] == "success"
    try:
        assert sim.set_joint_positions({"1": 0.3})["status"] == "success"
        assert sim.set_joint_positions({"shoulder_pan": 0.1})["status"] == "success"
    finally:
        sim.cleanup()


def test_unknown_key_keeps_the_not_a_joint_refusal(two_arms):
    r = two_arms.set_joint_positions({"nope": 0.3})
    assert r["status"] == "error"
    assert "keys are not joints in this model" in _text(r)


def test_joint_key_owners_follow_attachment_order(two_arms):
    assert two_arms._joint_key_owners("1") == ["so101", "arm2"]
    assert two_arms._joint_key_owners("shoulder_pan") == ["so101", "arm2"]
    assert two_arms._joint_key_owners("so101/1") == []
    assert two_arms._joint_key_owners("nope") == []
