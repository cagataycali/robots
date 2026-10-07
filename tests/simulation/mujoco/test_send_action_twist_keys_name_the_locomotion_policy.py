"""A ``send_action`` naming only a base twist names the ``run_policy`` route.

``vx`` / ``vy`` / ``vyaw`` are the intent a legged robot's onboard controller
takes on hardware, and ``docs/learn/hardware/microduck.md`` teaches exactly that
call. In simulation no actuator carries a twist: the robot's locomotion policy
does, as ``target_velocity``. The refusal used to list the joints and stop; it
now names the policy the registry declares for the robot, and only when every
refused key is a twist component, so a joint typo keeps the plain refusal.
"""

import pytest

from strands_robots.registry import get_robot, list_robots
from strands_robots.registry.policies import list_policy_providers
from strands_robots.simulation.mujoco.simulation import Simulation

ARM_XML = """
<mujoco model="arm">
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <joint name="hip" type="hinge" axis="0 1 0" range="-1 1"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.03"/>
    </body>
  </worldbody>
  <actuator>
    <position name="a_hip" joint="hip" kp="20" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""


@pytest.mark.parametrize(
    ("data_config", "action", "provider"),
    [
        pytest.param("microduck", {"vx": 0.1, "vyaw": 0.0}, "microduck", id="microduck-twist"),
        pytest.param("g1", {"vx": 0.1}, "wbc", id="g1-twist"),
        pytest.param("microduck", {"vx": 0.1, "a_hpi": 0.0}, None, id="twist-beside-a-typo"),
        pytest.param("microduck", {"left_hip_zyx": 0.0}, None, id="joint-typo"),
        pytest.param("so101", {"vx": 0.1}, None, id="no-locomotion-policy"),
    ],
)
def test_refusal_names_the_locomotion_policy_only_for_a_pure_twist(tmp_path, data_config, action, provider):
    xml = tmp_path / "arm.xml"
    xml.write_text(ARM_XML)
    sim = Simulation()
    sim.create_world()
    sim.add_robot(name="bot", urdf_path=str(xml), data_config=data_config)
    try:
        result = sim.send_action(action, robot_name="bot")
    finally:
        sim.cleanup()
    text = result["content"][0]["text"]
    assert result["status"] == "error"
    assert result["content"][1]["json"]["applied"] == []
    if provider is None:
        assert "target_velocity" not in text
    else:
        assert f"policy_provider='{provider}'" in text
        assert "policy_kwargs={'target_velocity': [vx, vy, vyaw]}" in text


def test_every_declared_locomotion_policy_is_a_registered_provider():
    names = [robot["name"] for robot in list_robots()]
    declared = {(get_robot(name) or {}).get("locomotion_policy") for name in names} - {None}
    assert declared, "no robot declares a locomotion_policy, so the hint can never fire"
    assert declared <= set(list_policy_providers())
