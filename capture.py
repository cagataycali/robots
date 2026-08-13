"""Measure what an Isaac articulation receives when a primitive runs under a policy.

The policy loop and the primitive both write the articulation's PD position
targets. This drives the REAL primitive with the REAL preamble/abort helpers
against the test suite's articulation fake, records the joint state each party
left behind, and replays those poses onto a MuJoCo arm carrying the same joint
names so the outcome is visible rather than tabular.
"""
from __future__ import annotations

import json
import pathlib
import sys
import types

import numpy as np

import strands_robots

ROOT = pathlib.Path(strands_robots.__file__).parents[1]
print("TREE:", ROOT)
sys.path.insert(0, str(ROOT))

from tests.simulation.isaac.test_motion_primitives import (  # noqa: E402
    _FakeArticulation,
    _FakeArticulationAction,
    _FakeWorld,
)
from strands_robots.simulation.isaac.simulation import IsaacSimulation, _RobotState  # noqa: E402

for name in ("isaacsim", "isaacsim.core", "isaacsim.core.utils", "isaacsim.core.utils.types"):
    sys.modules[name] = types.ModuleType(name)
sys.modules["isaacsim.core.utils.types"].ArticulationAction = _FakeArticulationAction

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_roll", "jaw"]
LIMITS: list[tuple[float, float] | None] = [(-3.1, 3.1), (-1.8, 1.8), (-2.4, 2.4), (-1.7, 1.7), (-0.2, 1.5)]
HOME = [0.0, 0.0, 0.0, 0.0, 0.0]
# What the rollout is driving toward - a reach-and-close trajectory.
POLICY_GOAL = [1.10, -1.05, 1.55, -0.95, 1.30]
POLICY_TICKS = 30
PRIMITIVE_AT = 10
WRIST_GOAL = 1.55  # what rotate_wrist would command instead


def run(with_primitive: bool) -> dict:
    art = _FakeArticulation(JOINTS, LIMITS, positions=list(HOME), servo_rate=0.5)
    state = {"tick": 0}

    traj: list[list[float]] = []

    def hook() -> None:
        # The policy loop's own write for this tick: a linear ramp to its goal.
        traj.append([float(v) for v in art.positions])
        k = min(state["tick"] + 1, POLICY_TICKS)
        f = k / POLICY_TICKS
        art.apply_action(
            _FakeArticulationAction(
                joint_positions=[h + f * (g - h) for h, g in zip(HOME, POLICY_GOAL, strict=True)],
                joint_indices=list(range(len(JOINTS))),
            )
        )
        state["tick"] += 1

    sim = IsaacSimulation()
    sim._world = _FakeWorld(art, on_step=hook)
    sim._world_created = True
    sim._robots["arm"] = _RobotState(
        name="arm", prim_path="/World/Robots/arm", joint_names=list(JOINTS), articulation=art
    )
    sim._robots["arm"].policy_running = True

    for _ in range(PRIMITIVE_AT):  # the rollout is already running
        sim._world.step()
    result = None
    if with_primitive:
        result = sim.rotate_wrist(robot_name="arm", target_yaw=WRIST_GOAL, max_steps=20)
    while state["tick"] < POLICY_TICKS:  # the rollout carries on
        sim._world.step()

    return {
        "primitive_status": None if result is None else result["status"],
        "primitive_text": None if result is None else result["content"][0].get("text", ""),
        "primitive_refused": bool(result is not None and result["status"] == "error"),
        "policy_ticks": state["tick"],
        "world_steps": sim._world.steps,
        "final_positions": [float(v) for v in art.positions],
        "target_writes": len(art.applied),
        "trajectory": traj,
    }


def render(poses: dict[str, list[float]]) -> dict[str, str]:
    """Replay each recorded joint state onto a MuJoCo arm and render it."""
    import imageio.v3 as iio

    from strands_robots import Simulation

    xml = """
<mujoco model="arm5">
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <visual><headlight ambient="0.55 0.55 0.55" diffuse="0.7 0.7 0.7"/><global offwidth="1600" offheight="1200"/></visual>
  <worldbody>
    <light pos="0.4 -0.4 0.9" dir="-0.4 0.4 -0.9"/>
    <body name="base" pos="0 0 0.03">
      <geom type="cylinder" size="0.045 0.03" rgba="0.30 0.32 0.36 1"/>
      <body name="l1" pos="0 0 0.03">
        <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-3.1 3.1" damping="3"/>
        <joint name="shoulder_lift" type="hinge" axis="0 1 0" range="-1.8 1.8" damping="3"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.17" size="0.026" rgba="0.22 0.52 0.85 1"/>
        <body name="l2" pos="0 0 0.17">
          <joint name="elbow" type="hinge" axis="0 1 0" range="-2.4 2.4" damping="3"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.15" size="0.023" rgba="0.30 0.66 0.42 1"/>
          <body name="l3" pos="0 0 0.15">
            <joint name="wrist_roll" type="hinge" axis="0 1 0" range="-1.7 1.7" damping="2"/>
            <geom type="capsule" fromto="0 0 0 0 0 0.07" size="0.019" rgba="0.92 0.62 0.16 1"/>
            <body name="jaw_b" pos="0 0 0.07">
              <joint name="jaw" type="hinge" axis="1 0 0" range="-0.2 1.5" damping="1"/>
              <geom type="box" size="0.012 0.030 0.014" rgba="0.86 0.24 0.24 1"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a1" joint="shoulder_pan" kp="60"/>
    <position name="a2" joint="shoulder_lift" kp="60"/>
    <position name="a3" joint="elbow" kp="50"/>
    <position name="a4" joint="wrist_roll" kp="40"/>
    <position name="a5" joint="jaw" kp="20"/>
  </actuator>
</mujoco>
"""
    out: dict[str, str] = {}
    d = pathlib.Path("/tmp/art-%s" % ROOT.name)
    d.mkdir(exist_ok=True)
    for label, qpos in poses.items():
        sim = Simulation(backend="mujoco", mesh=False)
        try:
            assert sim.create_world(ground_plane=False)["status"] == "success"
            p = pathlib.Path("/tmp/arm5-%s.xml" % ROOT.name)
            p.write_text(xml)
            assert sim.add_robot(name="arm", urdf_path=str(p))["status"] == "success"
            assert (
                sim.add_camera(name="look", position=[0.22, -0.24, 0.34], target=[0.00, 0.02, 0.26], fov=40)[
                    "status"
                ]
                == "success"
            )
            assert sim.set_joint_positions(dict(zip(JOINTS, qpos, strict=True)))["status"] == "success"
            r = sim.render(camera_name="look", width=720, height=660)
            assert r.get("status") == "success", r
            png = next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)
            fp = d / f"{label}.png"
            fp.write_bytes(png)
            img = np.asarray(iio.imread(fp))
            sat = float(((img.max(2).astype(int) - img.min(2)) > 45).mean())
            print(f"  {label:10s} saturated={sat:.4f}")
            assert sat > 0.02, (label, sat)
            out[label] = str(fp)
        finally:
            sim.cleanup()
    return out


facts: dict = {"tree": str(ROOT)}
facts["reference"] = run(with_primitive=False)
facts["with_primitive"] = run(with_primitive=True)
# The two command streams in conflict, rendered as poses. The rollout drives
# wrist_roll one way and rotate_wrist drives the same joint the other way; on
# main both write for 20 ticks. Which one a given tick resolves to is arbitrary
# in a real race, so the panels show the two COMMANDS rather than asserting an
# outcome the articulation fake's last-writer-wins servo cannot decide.
rollout_pose = facts["reference"]["final_positions"]
primitive_pose = list(rollout_pose)
primitive_pose[JOINTS.index("wrist_roll")] = WRIST_GOAL
facts["renders"] = render({"rollout": rollout_pose, "primitive": primitive_pose})
facts["rollout_pose"] = rollout_pose
facts["primitive_pose"] = primitive_pose
facts["wrist_goal"] = WRIST_GOAL
facts["policy_goal"] = POLICY_GOAL
facts["trajectory_identical_to_reference"] = (
    facts["with_primitive"]["trajectory"] == facts["reference"]["trajectory"]
)
facts["contended_writes"] = facts["with_primitive"]["target_writes"] - facts["reference"]["target_writes"]
out = pathlib.Path("/tmp/facts-%s.json" % ROOT.name)
out.write_text(json.dumps(facts, indent=2))
print(json.dumps({k: v for k, v in facts.items() if k != "renders"}, indent=2))
print("WROTE", out)
