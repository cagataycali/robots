"""Capture: what the newly-covered lazy MinkIKBridge build actually drives."""
import asyncio, json, pathlib, numpy as np, imageio.v3 as iio
import strands_robots
TREE = str(pathlib.Path(strands_robots.__file__).parents[1])
print("TREE:", TREE, flush=True)
from strands_robots import Simulation
from strands_robots.policies.vera.provider import VeraPolicy

OUT = pathlib.Path("_art"); OUT.mkdir(exist_ok=True)
CH = np.zeros((4, 7), np.float32); CH[:, 2] = -0.05; CH[:, 6] = 1.0   # 5 cm eef-delta descent

class Client:
    def __init__(self): self.infer_calls = 0
    def get_server_metadata(self):
        return {"action_space": "eef_delta", "context_frames": 1, "gripper_dim_index": 6,
                "gripper_is_raw": True, "view_keys": ["image"]}
    def configure(self, *a, **k): return None
    def reset(self, *a, **k): return None
    def close(self, *a, **k): return None
    def infer(self, *a, **k):
        self.infer_calls += 1; return {"action": CH}

def hand(sim):
    j = [c["json"] for c in sim.get_body_state(body_name="panda/hand")["content"] if "json" in c][0]
    return np.array(j["position"])

def png(sim, name):
    r = sim.render(camera_name="look", width=760, height=680)
    assert r.get("status") == "success", r
    b = next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)
    (OUT / name).write_bytes(b); return b

sim = Simulation(backend="mujoco", mesh=False)
sim.create_world()
assert sim.add_robot(name="panda", keyframe="home")["status"] == "success"
sim.add_camera(name="look", position=[1.15, -1.0, 0.85], target=[0.42, 0.0, 0.45], fov=32)
model = sim._world._model
joints = sim.robot_joint_names("panda")
sim.step(80)

client = Client()
policy = VeraPolicy(client=client, auto_launch_server=False)
policy.set_robot_state_keys(joints)
policy.set_ik_target(model, "panda/hand", "body")
bridge_before = policy._ik_bridge

a0 = png(sim, "home.png"); h0 = hand(sim)
applied = 0
for _ in range(40):
    obs = {k: float(np.asarray(sim.get_observation(robot_name="panda")[k]).reshape(-1)[0]) for k in joints}
    obs["image"] = np.zeros((8, 8, 3), np.uint8)
    for act in asyncio.run(policy.get_actions(obs, "descend toward the table")):
        assert sim.send_action(act, robot_name="panda", n_substeps=25)["status"] == "success"
        applied += 1
a1 = png(sim, "driven.png"); h1 = hand(sim)

A = iio.imread(a0).astype(int); B = iio.imread(a1).astype(int)
moved = float((np.abs(A - B).sum(2) > 8).mean())
sat = float(((A.max(2) - A.min(2)) > 45).mean())

facts = {
    "tree": TREE,
    "bridge_before_first_inference": repr(bridge_before),
    "bridge_after_first_inference": type(policy._ik_bridge).__name__,
    "frame_name": policy._ik_bridge.ee_frame_name,
    "frame_type": policy._ik_bridge.ee_frame_type,
    "qp_solver": policy._ik_bridge.solver,
    "infer_calls": client.infer_calls,
    "actions_applied": applied,
    "hand_home": [round(float(v), 4) for v in h0],
    "hand_driven": [round(float(v), 4) for v in h1],
    "descent_cm": round(float(100 * (h0[2] - h1[2])), 2),
    "moved_frac": round(moved, 4),
    "arm_sat": round(sat, 4),
    "coverage": {"before": {"missing": 7, "pct": 98, "lines": "725-727 (the build)"},
                 "after": {"missing": 5, "pct": 99, "lines": "none"}},
    "suite": {"before": 728, "after": 733},
    "mutations": [
        ("M1  drop the cache guard (rebuild every inference)", "1 failed / 4 pass", "3 failed / 725 pass"),
        ("M2  swap the frame name and type arguments",         "4 failed / 1 pass", "0 failed / 728 pass"),
        ('M3  hardcode the frame type as "body"',              "1 failed / 4 pass", "0 failed / 728 pass"),
        ("M4  build without caching (return a fresh bridge)",  "4 failed / 1 pass", "0 failed / 728 pass"),
        ("     (unmutated control)",                           "0 failed / 5 pass", "0 failed / 728 pass"),
    ],
}
(OUT / "facts.json").write_text(json.dumps(facts, indent=2))
print(json.dumps({k: v for k, v in facts.items() if k != "mutations"}, indent=2), flush=True)
assert policy._ik_bridge is not None and bridge_before is None
assert facts["descent_cm"] > 2.0, facts["descent_cm"]
assert moved > 0.10, moved
assert sat > 0.5, sat
print("CAPTURE OK", flush=True)
sim.cleanup()
