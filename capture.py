"""Capture a real MuJoCo evaluation on this tree. Prints its own tree first."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

import strands_robots.simulation.policy_runner as runner_mod

TREE = str(Path(runner_mod.__file__).parents[2])
print("TREE:", TREE, flush=True)

from strands_robots.policies import MockPolicy  # noqa: E402
from strands_robots.simulation import Simulation  # noqa: E402
from strands_robots.simulation.policy_runner import PolicyRunner  # noqa: E402

ARM = """<mujoco model="probe_arm">
  <compiler angle="radian"/>
  <visual><global offwidth="1600" offheight="1200"/>
    <headlight ambient="0.55 0.55 0.55" diffuse="0.65 0.65 0.65"/>
  </visual>
  <worldbody>
    <body name="base" pos="0 0 0.04">
      <geom type="cylinder" size="0.05 0.04" rgba="0.30 0.32 0.36 1"/>
      <body name="link1" pos="0 0 0.04">
        <joint name="shoulder" type="hinge" axis="0 1 0" range="-1.4 1.4" damping="3"/>
        <geom type="capsule" fromto="0 0 0 0.30 0 0" size="0.030" rgba="0.20 0.45 0.85 1"/>
        <body name="link2" pos="0.30 0 0">
          <joint name="elbow" type="hinge" axis="0 1 0" range="-1.4 1.4" damping="3"/>
          <geom type="capsule" fromto="0 0 0 0.24 0 0" size="0.025" rgba="0.95 0.55 0.12 1"/>
          <body name="tip" pos="0.24 0 0">
            <geom type="sphere" size="0.042" rgba="0.15 0.75 0.35 1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a_shoulder" joint="shoulder" kp="40" ctrlrange="-1.4 1.4"/>
    <position name="a_elbow" joint="elbow" kp="40" ctrlrange="-1.4 1.4"/>
  </actuator>
</mujoco>
"""

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
TAG = Path(TREE).name


def png(sim: Any, name: str) -> str:
    res = sim.render(camera_name="look", width=560, height=440)
    assert res.get("status") == "success", res
    data = next(b["image"]["source"]["bytes"] for b in res["content"] if "image" in b)
    p = OUT / f"{name}.png"
    p.write_bytes(data)
    return str(p)


def build() -> tuple[Any, Any]:
    sim = Simulation(backend="mujoco", tool_name="art", mesh=False)
    assert sim.create_world()["status"] == "success"
    xml = OUT / "arm.xml"
    xml.write_text(ARM)
    assert sim.add_robot(name="arm", urdf_path=str(xml))["status"] == "success"
    # Before the rollout: add_camera recompiles the spec and drops ctrl.
    assert sim.add_camera(name="look", position=[0.30, -1.05, 0.55], target=[0.28, 0, 0.16], fov=38)[
        "status"
    ] == "success"
    policy = MockPolicy()
    policy.set_robot_state_keys(sim.robot_action_keys("arm"))
    return sim, policy


def one(label: str, **bounds: Any) -> dict[str, Any]:
    sim, policy = build()
    before = png(sim, f"{TAG}_{label}_before")
    applied = {"n": 0}
    real_send = sim.send_action

    def counting(*a: Any, **k: Any) -> Any:
        applied["n"] += 1
        return real_send(*a, **k)

    sim.send_action = counting  # type: ignore[method-assign]
    row: dict[str, Any] = {"case": label, **{k: repr(v) for k, v in bounds.items()}}
    try:
        res = PolicyRunner(sim).evaluate(
            "arm", policy, success_fn=lambda obs: False, control_frequency=50.0, **bounds
        )
        j: dict[str, Any] = {}
        for b in res.get("content", []):
            if "json" in b:
                j = b["json"]
        row.update(
            status=res.get("status"),
            episodes=j.get("episodes_completed"),
            rate=j.get("success_rate"),
            measured=j.get("success_measured"),
            avg_steps=j.get("avg_steps"),
        )
    except ValueError as exc:
        row.update(status="refused", message=str(exc))
    row["applied"] = applied["n"]
    obs = sim.get_observation(robot_name="arm")
    row["joints"] = {k: round(float(v), 6) for k, v in obs.items() if not hasattr(v, "shape")}
    row["after"] = png(sim, f"{TAG}_{label}_after")
    row["before"] = before
    sim.cleanup()
    return row


rows = [
    one("honored", n_episodes=2, max_steps=60),
    one("max_steps_0", n_episodes=2, max_steps=0),
    one("n_episodes_0", n_episodes=0, max_steps=60),
]
(OUT / f"facts_{TAG}.json").write_text(json.dumps({"tree": TREE, "rows": rows}, indent=1, default=str))
print(json.dumps({"tree": TREE, "rows": [{k: v for k, v in r.items() if "png" not in str(v)} for r in rows]},
                 indent=1, default=str), flush=True)
