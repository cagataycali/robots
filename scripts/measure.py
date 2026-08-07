"""Measure SimEnv(action_scale=...) end to end on a real MuJoCo arm."""
from __future__ import annotations
import json, math, pathlib, sys

import strands_robots.training.rl.env as envmod
print("TREE:", pathlib.Path(envmod.__file__).parents[3], flush=True)

import numpy as np
import torch
from strands_robots.simulation import Simulation
from strands_robots.training.rl import SimEnv

ARM = """<mujoco model="probe">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.05 0.05" rgba="0.3 0.3 0.35 1"/>
      <body name="link1" pos="0 0 0.06">
        <joint name="shoulder" type="hinge" axis="0 0 1" range="-2.0 2.0" limited="true" damping="4"/>
        <geom type="capsule" fromto="0 0 0 0.28 0 0" size="0.03" rgba="0.85 0.45 0.1 1"/>
        <body name="link2" pos="0.28 0 0">
          <joint name="elbow" type="hinge" axis="0 1 0" range="-2.0 2.0" limited="true" damping="4"/>
          <geom type="capsule" fromto="0 0 0 0.22 0 0" size="0.026" rgba="0.2 0.5 0.85 1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a_shoulder" joint="shoulder" kp="50" ctrlrange="-2 2"/>
    <position name="a_elbow" joint="elbow" kp="50" ctrlrange="-2 2"/>
  </actuator>
</mujoco>
"""

CASES = [
    ("1.0 (default)", 1.0),
    ("0.5", 0.5),
    ("0", 0),
    ("0.0", 0.0),
    ("-1.0", -1.0),
    ("nan", float("nan")),
    ("inf", float("inf")),
    ("-inf", float("-inf")),
    ("True", True),
    ("'0.5'", "0.5"),
    ("None", None),
    ("[1.0]", [1.0]),
]

def build(tmp: pathlib.Path):
    sim = Simulation(backend="mujoco", tool_name="probe", mesh=False)
    sim.create_world()
    p = tmp / "arm.xml"
    p.write_text(ARM)
    sim.add_robot(name="arm", urdf_path=str(p))
    return sim

def run_case(label, value, tmp):
    sim = build(tmp)
    rec = {"label": label, "repr": repr(value)}
    try:
        env = SimEnv(
            sim,  # type: ignore[arg-type]
            actor_obs_keys=["shoulder", "elbow"],
            reward_terms=[lambda e: 1.0],
            robot_name="arm",
            action_scale=value,  # type: ignore[arg-type]
            max_episode_steps=1000,
            n_substeps=10,
        )
    except BaseException as e:  # noqa: BLE001 - classify the ctor outcome
        rec.update(ctor="raised", exc=f"{type(e).__name__}: {e}")
        sim.cleanup()
        return rec
    rec["ctor"] = "accepted"
    rec["stored"] = repr(env.action_scale)

    statuses: list[str] = []
    real_send = sim.send_action
    def spy(action, robot_name=None, n_substeps=1):
        r = real_send(action, robot_name=robot_name, n_substeps=n_substeps)
        statuses.append(r.get("status", "?"))
        return r
    sim.send_action = spy  # type: ignore[method-assign]

    env.reset()
    total = 0.0
    cmd = torch.tensor([0.9, -0.7], dtype=torch.float32)
    step_exc = None
    try:
        for _ in range(60):
            _o, r, _d, _i = env.step(cmd)
            total += float(r.item())
    except BaseException as e:  # noqa: BLE001
        step_exc = f"{type(e).__name__}: {e}"
    obs = sim.get_observation(robot_name="arm", skip_images=True)
    rec.update(
        step_exc=step_exc,
        n_send=len(statuses),
        n_ok=statuses.count("success"),
        n_err=sum(1 for s in statuses if s != "success"),
        reward=round(total, 3),
        shoulder=round(float(obs["shoulder"]), 6),
        elbow=round(float(obs["elbow"]), 6),
    )
    sim.cleanup()
    return rec

def main() -> None:
    tmp = pathlib.Path("/tmp/probe_arm"); tmp.mkdir(exist_ok=True)
    out = [run_case(l, v, tmp) for l, v in CASES]
    print(json.dumps(out, indent=1))
    print("\n=== TABLE ===")
    hdr = f"{'action_scale':<15} {'ctor':<9} {'stored':<10} {'sends':<6} {'ok':<4} {'err':<4} {'reward':<8} {'shoulder':<11} {'elbow':<11}"
    print(hdr); print("-" * len(hdr))
    for r in out:
        if r["ctor"] == "raised":
            print(f"{r['label']:<15} RAISED    {r['exc'][:70]}")
        else:
            print(f"{r['label']:<15} {r['ctor']:<9} {r['stored']:<10} {r['n_send']:<6} {r['n_ok']:<4} {r['n_err']:<4} {r['reward']:<8} {r['shoulder']:<11} {r['elbow']:<11}")

main()
