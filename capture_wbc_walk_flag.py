"""Capture what a caller asking for the WBC balance controller actually gets."""
import json, sys
from pathlib import Path
import numpy as np
import strands_robots
from strands_robots import Robot
from strands_robots.registry.policies import build_policy_kwargs
from strands_robots.policies import create_policy

TREE = Path(strands_robots.__file__).parents[1]
print("TREE:", TREE, flush=True)
out_dir = Path(sys.argv[1]); out_dir.mkdir(parents=True, exist_ok=True)
CKPT = "/home/cagatay/.cache/huggingface/hub/models--nepyope--GR00T-WholeBodyControl_g1/snapshots/7bb8a672f5a4213c9261ea5ac1f3f034f5078638"
DURATION, HZ = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0, 50.0

# The public registry helper under change, asked for the NON-walking controller.
kwargs = build_policy_kwargs(
    "wbc", checkpoint=CKPT, walk=False, target_velocity=[0.6, 0.0, 0.0]
)
print("build_policy_kwargs(wbc, checkpoint=..., walk=False, target_velocity=[0.6,0,0]) ->",
      {k: (v if k != "checkpoint" else "<ckpt>") for k, v in kwargs.items()}, flush=True)

policy = create_policy("wbc", **kwargs)
# Which ONNX session actually ran is the exact binary consequence: ``walk=False``
# is documented as "only the main policy", so the walk session must never run.
runs = {"main": 0, "walk": 0}
def _count(sess, tag):
    if sess is None:
        return None
    real = sess.run
    def wrapped(*a, **kw):
        runs[tag] += 1
        return real(*a, **kw)
    sess.run = wrapped
    return sess
policy._load_models() if not policy.policy_session else None
_count(policy.policy_session, "main")
_count(policy.walk_session, "walk")
sim = Robot("unitree_g1", mesh=False)
# A static start-line marker so translation is readable off the frame.
sim.add_object(name="startpost", shape="box", size=[0.09, 0.09, 0.80],
               position=[0.0, -0.95, 0.40], color=[0.95, 0.35, 0.08, 1.0], is_static=True)
# A 1 m ruler *beside* the corridor, so how far the robot walked is readable
# without placing anything the feet could contact.
for _m in (1, 2, 3, 4):
    sim.add_object(name=f"post{_m}", shape="box", size=[0.07, 0.07, 0.55],
                   position=[float(_m), -0.95, 0.275],
                   color=[0.20, 0.45, 0.90, 1.0], is_static=True)
sim.add_camera(name="side", position=[1.85, -7.6, 2.10], target=[1.85, 0.0, 0.72], fov=45)

W, H = 640, 480
frames, trace = [], []
real_get_obs = sim.get_observation
n = {"i": 0}
def spy(*a, **kw):
    obs = real_get_obs(*a, **kw)
    i = n["i"]; n["i"] += 1
    if i % 25 == 0:                       # every 0.5 s
        r = sim.render(camera_name="side", width=W, height=H)
        png = next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)
        frames.append((i / HZ, png))
    st = sim.get_body_state(body_name="pelvis")
    j = next(c["json"] for c in st["content"] if "json" in c)
    trace.append([i / HZ, *[float(v) for v in j["position"]]])
    return obs
sim.get_observation = spy

res = sim.run_policy(robot_name="unitree_g1", policy_object=policy,
                     duration=DURATION, control_frequency=HZ, action_horizon=1)
sim.get_observation = real_get_obs
status = res.get("status")
print("run_policy status:", status, flush=True)

arr = np.array(trace)
facts = {
    "tree": str(TREE),
    "kwargs_walk": kwargs.get("walk"),
    "policy_walk_flag": bool(policy._walk),
    "walk_session_loaded": policy.walk_session is not None,
    "main_session_runs": runs["main"],
    "walk_session_runs": runs["walk"],
    "status": status,
    "n_ticks": int(arr.shape[0]),
    "x_start": float(arr[0, 1]), "x_end": float(arr[-1, 1]),
    "travel_x": float(abs(arr[-1, 1] - arr[0, 1])),
    "z_start": float(arr[0, 3]), "z_end": float(arr[-1, 3]),
    "frames": [],
}
for t, png in frames:
    fp = out_dir / f"f_{t:05.2f}.png"; fp.write_bytes(png)
    facts["frames"].append({"t": t, "path": str(fp)})
np.save(out_dir / "trace.npy", arr)
(out_dir / "facts.json").write_text(json.dumps(facts, indent=2))
print(json.dumps({k: v for k, v in facts.items() if k != "frames"}, indent=2))
