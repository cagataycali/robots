"""Measure cleanup's join for each candidate budget, plus a real rollout render.

Run in a worktree at upstream/main and in the branch; every number in the figure
comes from these dumps.
"""

import json
import logging
import math
import pathlib
import sys
import time

import numpy as np

import strands_robots.simulation.mujoco.simulation as M

TREE = str(pathlib.Path(M.__file__).parents[3])
print("TREE:", TREE, flush=True)

from strands_robots import Simulation  # noqa: E402

ARM_XML = """<mujoco model="budget">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81"/>
  <visual><headlight ambient="0.55 0.55 0.55" diffuse="0.65 0.65 0.65"/></visual>
  <worldbody>
    <light pos="0.6 -0.6 1.2" dir="-0.4 0.4 -1"/>
    <geom name="floor" type="plane" size="2 2 0.05" rgba="0.82 0.83 0.86 1"/>
    <body name="base" pos="0 0 0.03">
      <geom type="box" size="0.06 0.06 0.03" rgba="0.30 0.32 0.38 1"/>
      <body name="link1" pos="0 0 0.05">
        <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="4"/>
        <geom type="capsule" fromto="0 0 0 0.24 0 0" size="0.028" rgba="0.95 0.55 0.12 1"/>
        <body name="link2" pos="0.24 0 0">
          <joint name="elbow" type="hinge" axis="0 1 0" range="-2 2" damping="3"/>
          <geom type="capsule" fromto="0 0 0 0.20 0 0" size="0.023" rgba="0.16 0.55 0.90 1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a_shoulder" joint="shoulder" kp="30" ctrlrange="-2 2"/>
    <position name="a_elbow" joint="elbow" kp="25" ctrlrange="-2 2"/>
  </actuator>
</mujoco>
"""

XML = pathlib.Path("/tmp/budget_arm.xml")
XML.write_text(ARM_XML)


class Cap(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.msgs: list[str] = []
        self.dropped = 0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.msgs.append(record.getMessage())
        except Exception:
            self.dropped += 1

    def handleError(self, record: logging.LogRecord) -> None:
        self.dropped += 1


class _Sentinel:
    pass


OMITTED = _Sentinel()


def one(budget, tag):
    """Start a live 50 Hz rollout, cleanup with ``budget``, report the outcome."""
    sim = Simulation(tool_name=f"b{tag}", mesh=False)
    sim.create_world()
    sim.add_robot(name="arm", urdf_path=str(XML))
    cap = Cap()
    lg = M.logger
    lg.addHandler(cap)
    prev_level, prev_prop = lg.level, lg.propagate
    lg.setLevel(logging.DEBUG)
    lg.propagate = False
    sim.start_policy(robot_name="arm", policy_provider="mock", duration=20.0, control_frequency=50.0)
    time.sleep(0.5)
    futs = list(sim._policy_threads.values())
    t0 = time.perf_counter()
    raised = None
    try:
        sim.cleanup() if budget is OMITTED else sim.cleanup(policy_stop_timeout=budget)
    except BaseException as exc:  # noqa: BLE001 - the outcome under measurement
        raised = f"{type(exc).__name__}: {exc}"
    waited = time.perf_counter() - t0
    abandoned = any(not f.done() for f in futs)
    lg.removeHandler(cap)
    lg.setLevel(prev_level)
    lg.propagate = prev_prop
    time.sleep(0.25)
    return {
        "label": "<omitted>" if budget is OMITTED else repr(budget),
        "waited_s": round(waited, 4),
        "worker_abandoned": bool(abandoned),
        "world_released": sim._world is None,
        "reported": bool([m for m in cap.msgs if "policy_stop_timeout" in m or "did not stop" in m]),
        "dropped_records": cap.dropped,
        "raised": raised,
    }


def render_reference():
    """A normal rollout to completion, rendered - the path that must not change."""
    sim = Simulation(tool_name="ref", mesh=False)
    sim.create_world()
    sim.add_robot(name="arm", urdf_path=str(XML))
    sim.add_camera(name="look", position=[0.66, -0.62, 0.46], target=[0.16, 0.0, 0.14], fov=40)
    sim.run_policy(robot_name="arm", policy_provider="mock", duration=1.2, control_frequency=50.0)
    obs = sim.get_observation(robot_name="arm")
    joints = {k: round(float(v), 6) for k, v in obs.items() if not hasattr(v, "shape")}
    res = sim.render(camera_name="look", width=680, height=560)
    png = next(c["image"]["source"]["bytes"] for c in res["content"] if "image" in c)
    sim.cleanup(policy_stop_timeout=2.0)
    return joints, png


def main() -> None:
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    one(0.5, "warm")  # the first worker in a process ignores the cooperative stop
    rows = [one(b, i) for i, b in enumerate([OMITTED, 2.0, math.inf, math.nan, 0.0, -1.0, True, "5", np.float32(0.25)])]
    joints, png = render_reference()
    (out / "reference.png").write_bytes(png)
    (out / "facts.json").write_text(json.dumps({"tree": TREE, "rows": rows, "reference_joints": joints}, indent=2))
    print(json.dumps({"tree": TREE, "rows": rows, "reference_joints": joints}, indent=2), flush=True)


if __name__ == "__main__":
    main()
