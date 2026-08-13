"""Capture: the mount the Isaac refusal redirects to, on a backend that has it."""
import json
import os
import pathlib

import numpy as np
import strands_robots

ROOT = pathlib.Path(strands_robots.__file__).parents[1]
print("TREE:", ROOT)
OUT = pathlib.Path("/tmp/art-" + os.environ["GITHUB_RUN_ID"])
OUT.mkdir(exist_ok=True)
facts = {"tree": str(ROOT)}


def save(): (OUT / "facts.json").write_text(json.dumps(facts, indent=2))


def png(sim, cam, w=520, h=440):
    r = sim.render(camera_name=cam, width=w, height=h)
    assert r.get("status") == "success", r
    b = next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)
    import imageio.v3 as iio
    return iio.imread(b)


# --- The Isaac answer (no Isaac Sim needed: the refusal precedes the lock) -----
from strands_robots.simulation.isaac.simulation import IsaacSimulation

sk = IsaacSimulation.__new__(IsaacSimulation)
r = IsaacSimulation.add_camera(sk, name="wrist", parent_body="so101/gripper")
facts["isaac_status"] = r["status"]
facts["isaac_text"] = r["content"][0]["text"]
print("isaac:", facts["isaac_status"])
save()

# --- The mount itself, on the backend the refusal names ----------------------
from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

sim = MuJoCoSimEngine(tool_name="art_mount", mesh=False)
try:
    assert sim.create_world()["status"] == "success"
    assert sim.add_robot(name="so101")["status"] == "success"
    m = sim.add_camera(name="wrist", parent_body="so101/gripper", position=[0.06, 0.0, 0.03],
                       target=[0.0, 0.0, 0.0], fov=70)
    facts["mujoco_mount_status"] = m["status"]
    facts["mujoco_mount_text"] = m["content"][0]["text"]
    print("mujoco mount:", facts["mujoco_mount_status"])
    save()

    keys = sim.robot_action_keys("so101")
    facts["action_keys"] = list(keys)

    # Frame A: the arm at rest.
    frames = {}
    obs = sim.get_observation(robot_name="so101")
    frames["rest"] = png(sim, "wrist")
    facts["rest_joints"] = {k: round(float(v), 4) for k, v in sorted(obs.items())
                            if not hasattr(v, "shape")}

    # Drive the arm; a MOUNTED camera's view must change because it RIDES.
    # so101's actuators are named '1'..'6' - use the keys the robot reports, or
    # every command is dropped and the view changes only from gravity droop.
    ramp = dict(zip(keys, [1.10, -1.30, 1.70, 0.0, 0.0, 1.10], strict=False))
    facts["commanded"] = ramp
    applied_ok = 0
    for i in range(90):
        f = (i + 1) / 90.0
        r_a = sim.send_action({k: v * f for k, v in ramp.items()}, robot_name="so101", n_substeps=10)
        applied_ok += int(r_a.get("status") == "success")
    facts["actions_applied_ok"] = applied_ok
    assert applied_ok == 90, applied_ok
    obs2 = sim.get_observation(robot_name="so101")
    frames["moved"] = png(sim, "wrist")
    facts["moved_joints"] = {k: round(float(v), 4) for k, v in sorted(obs2.items())
                             if not hasattr(v, "shape")}

    a, b = frames["rest"].astype(int), frames["moved"].astype(int)
    diff = float((np.abs(a - b).max(2) > 8).mean())
    facts["wrist_view_changed_frac"] = round(diff, 4)
    sat = lambda im: float(((im.max(2).astype(int) - im.min(2)) > 45).mean())
    facts["rest_saturated"] = round(sat(frames["rest"]), 4)
    facts["moved_saturated"] = round(sat(frames["moved"]), 4)
    print(f"wrist view changed on {diff:.2%} of pixels (mount rides the body)")
    # A mounted camera's view MUST change when the body it rides moves.
    assert diff > 0.10, f"mounted wrist view barely changed ({diff:.2%}) - reframe"

    for k, v in frames.items():
        np.save(OUT / f"{k}.npy", v)
    save()
finally:
    sim.cleanup()
print("OK ->", OUT)
