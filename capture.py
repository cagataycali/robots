"""Measure what the new cases pin, and render the scene at each stage."""
from __future__ import annotations
import io, json, os, pathlib, sys, tempfile, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import strands_robots
print("TREE:", pathlib.Path(strands_robots.__file__).parents[1])
from PIL import Image
from strands_robots.policies.base import Policy
from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine
from strands_robots.simulation.recording import dataset_rate_mismatch_error

ARM = """<mujoco><visual><global offwidth="1600" offheight="1200"/></visual><worldbody><body name="l1">
<joint name="j1" type="hinge" axis="0 0 1" range="-1.5 1.5" damping="4"/>
<geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.025"/>
<body name="l2" pos="0.15 0 0">
<joint name="j2" type="hinge" axis="0 0 1" range="-1.5 1.5" damping="4"/>
<geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.025"/></body></body></worldbody>
<actuator><position name="a1" joint="j1" kp="30" ctrlrange="-1.5 1.5"/>
<position name="a2" joint="j2" kp="30" ctrlrange="-1.5 1.5"/></actuator></mujoco>"""

TMP = pathlib.Path(tempfile.mkdtemp(prefix="art-")); XML = TMP / "arm.xml"; XML.write_text(ARM)
OUT = pathlib.Path("_art"); W, H = 700, 620


class Reach(Policy):
    def __init__(self, keys): super().__init__(); self._k = list(keys)
    @property
    def provider_name(self): return "reach"
    @property
    def requires_images(self): return False
    def set_robot_state_keys(self, keys): pass
    async def get_actions(self, obs, instruction, **kw):
        return [{self._k[0]: 1.25, self._k[1]: -1.25}]


def engine(names=("arm",)):
    e = MuJoCoSimEngine(tool_name="art", mesh=False); e.create_world()
    for i, n in enumerate(names):
        e.add_robot(name=n, urdf_path=str(XML), position=[0.0, 0.65 * i, 0.0])
    e.add_camera(name="look", position=[0.22, -0.20, 0.20], target=[0.10, 0.0, 0.02], fov=40)
    return e


def shot(e):
    r = e.render(camera_name="look", width=W, height=H)
    assert r.get("status") == "success", r
    raw = next(b["image"]["source"]["bytes"] for b in r["content"] if "image" in b)
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.int16)


def txt(r):
    for b in (r or {}).get("content", []):
        if "text" in b: return b["text"]
    return "<none>"


def frames_on_disk(root):
    ps = [p for p in pathlib.Path(root).rglob("*.parquet") if "data" in p.parts]
    if not ps: return 0
    import pandas as pd
    return sum(len(pd.read_parquet(p)) for p in ps)


facts = {}

# ---- honored: start_policy with no recording open -> the rollout really runs
e = engine()
home = shot(e)
res = e.start_policy(robot_name="arm", policy_object=Reach(e.robot_action_keys("arm")),
                     duration=60.0, control_frequency=50.0)
assert res["status"] == "success", res
handle = e._world.robots["arm"]
deadline = time.monotonic() + 20.0
while not handle.policy_running:
    assert time.monotonic() < deadline, "worker never started"
    time.sleep(0.005)
time.sleep(1.6)
honored = shot(e)
facts["honored"] = {"status": res["status"], "text": txt(res)}
e.stop_policy(robot_name="arm")
fut = (e._policy_threads or {}).get("arm")
if fut is not None:
    try: fut.result(timeout=30.0)
    except Exception: pass
e.cleanup()

# ---- refused: start_policy against a 30 fps recording while capturing at 50
def joints(e, name="arm"):
    obs = e.get_observation(robot_name=name)
    return {k: round(float(v), 12) for k, v in sorted(obs.items()) if not hasattr(v, "shape")}


e = engine(); before_refuse = shot(e); q_before = joints(e)
root_a = str(TMP / "dsA")
assert e.start_recording(repo_id="local/dsA", task="hold", fps=30, root=root_a)["status"] == "success"
r = e.start_policy(robot_name="arm", policy_object=Reach(e.robot_action_keys("arm")),
                   duration=60.0, control_frequency=50.0)
facts["start_policy"] = {
    "status": r["status"], "text": txt(r),
    "policy_running": e._world.robots["arm"].policy_running,
    "threads": len(e._policy_threads or {}),
    "recorder_frames": e._active_recorder().frame_count,
    "frames_on_disk": frames_on_disk(root_a),
    "envelope_verbatim": r == dataset_rate_mismatch_error("start_policy", e._active_recorder(), 50.0),
}
after_refuse = shot(e); q_after = joints(e)
e.stop_recording(); e.cleanup()

# ---- refused: run_multi_policy, same disagreement
e = engine(("armA", "armB"))
root_b = str(TMP / "dsB")
assert e.start_recording(repo_id="local/dsB", task="hold", fps=30, root=root_b)["status"] == "success"
pol = {n: Reach(e.robot_action_keys(n)) for n in ("armA", "armB")}
r2 = e.run_multi_policy(policies=pol, n_steps=10, control_frequency=50.0)
facts["run_multi_policy"] = {
    "status": r2["status"], "text": txt(r2),
    "policy_running": [e._world.robots[n].policy_running for n in ("armA", "armB")],
    "threads": len(e._policy_threads or {}),
    "recorder_frames": e._active_recorder().frame_count,
    "frames_on_disk": frames_on_disk(root_b),
    "envelope_verbatim": r2 == dataset_rate_mismatch_error("run_multi_policy", e._active_recorder(), 50.0),
}
e.stop_recording(); e.cleanup()

# ---- relations the artifact asserts -----------------------------------------
def diff_frac(a, b): return float((np.abs(a - b).sum(axis=2) > 8).mean())
def sat_frac(a): return float(((a.max(axis=2) - a.min(axis=2)) > 45).mean())

moved = diff_frac(home, honored)
max_delta = int(np.abs(before_refuse - after_refuse).max())
changed_px = int((np.abs(before_refuse - after_refuse).sum(axis=2) > 8).sum())
facts["relations"] = {
    "honored_moved_frac": round(moved, 4),
    "refused_max_pixel_delta": max_delta,
    "refused_changed_pixels_over_threshold": changed_px,
    "total_pixels": int(before_refuse.shape[0] * before_refuse.shape[1]),
    "joints_identical_across_the_refusals": q_before == q_after,
    "home_sat_frac": round(sat_frac(home), 4),
}
assert moved > 0.10, f"honored rollout barely moves the arm: {moved:.4f}"
# max_delta 1 is renderer noise; the physics claim is the joint state and the
# count of pixels that changed by more than noise.
assert max_delta <= 2, max_delta
assert changed_px == 0, changed_px
assert q_before == q_after, (q_before, q_after)
assert facts["start_policy"]["status"] == "error" and facts["start_policy"]["threads"] == 0
assert facts["start_policy"]["policy_running"] is False
assert facts["run_multi_policy"]["status"] == "error" and facts["run_multi_policy"]["threads"] == 0
assert facts["start_policy"]["envelope_verbatim"] and facts["run_multi_policy"]["envelope_verbatim"]
assert facts["start_policy"]["frames_on_disk"] == 0 and facts["run_multi_policy"]["frames_on_disk"] == 0

np.save(OUT / "home.npy", home.astype(np.uint8))
np.save(OUT / "honored.npy", honored.astype(np.uint8))
np.save(OUT / "refused.npy", after_refuse.astype(np.uint8))
(OUT / "facts.json").write_text(json.dumps(facts, indent=2, default=str))
print(json.dumps(facts, indent=2, default=str))
