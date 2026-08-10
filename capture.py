"""Decode a VERA eef-delta chunk onto a real MuJoCo Panda, once per index spelling."""
import io
import json
import pathlib
import sys

import numpy as np
from PIL import Image

import strands_robots.policies.vera.sim_ik as sim_ik
from strands_robots.policies.vera.sim_ik import decode_vera_delta_chunk_to_targets
from strands_robots.simulation.ik import MinkIKBridge
from strands_robots import Simulation

TREE = pathlib.Path(sim_ik.__file__).parents[3]
print("TREE:", TREE)
OUT = pathlib.Path(sys.argv[1])
TAG = sys.argv[2]

# A 12-step descend: translation -Z, no rotation, gripper closing. Column 6 is
# the gripper; columns 0..5 are the pose block for rotation_dim=3.
STEPS = 12
chunk = np.zeros((STEPS, 7), dtype=np.float64)
chunk[:, 2] = -0.55          # descend (OSC-normalized)
chunk[:, 0] = 0.30           # and forward in +X so the move is legible
chunk[:, 6] = 1.0            # gripper column


def fresh():
    sim = Simulation(backend="mujoco", mesh=False)
    sim.create_world()
    sim.add_robot(name="panda")
    # Camera BEFORE any rollout: add_camera recompiles the spec and drops ctrl.
    sim.add_camera(name="look", position=[1.35, -1.05, 0.85], target=[0.35, 0.0, 0.35], fov=40)
    return sim


def render(sim):
    r = sim.render(camera_name="look", width=760, height=680)
    assert r.get("status") == "success", r
    png = next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))


def run(gidx):
    """Decode with this index and apply the final joint target to the arm."""
    sim = fresh()
    model = sim._world._model  # noqa: SLF001 - the artifact needs the compiled model
    names = sim.robot_joint_names("panda")
    jnames = names["joint_names"] if isinstance(names, dict) else names
    bridge = MinkIKBridge(model, "panda/hand", "body")
    q0 = np.asarray(sim._world._data.qpos[: model.nq], dtype=np.float64).copy()  # noqa: SLF001
    home = render(sim)
    verdict, detail, moved = "", "", None
    try:
        out = decode_vera_delta_chunk_to_targets(
            chunk, bridge, q0, rotation_dim=3, has_gripper=True, gripper_dim_index=gidx
        )
        qpos = np.asarray(out["qpos"])[-1]
        applied = {n: float(v) for n, v in zip(jnames, qpos[: len(jnames)], strict=False)}
        sim.set_joint_positions(applied)
        verdict = "decoded"
        detail = f"track err {out['tracking_error']['mean_mm']:.1f} mm mean"
        moved = render(sim)
    except Exception as exc:  # noqa: BLE001 - classifying every outcome is the point
        verdict = type(exc).__name__
        detail = str(exc).replace("\n", " ")[:150]
        moved = home
    hand = sim.get_body_state(body_name="panda/hand")
    pos = next(c["json"] for c in hand["content"] if "json" in c)["position"]
    sim.cleanup()
    return verdict, detail, np.asarray(pos, dtype=float), home, moved


CASES = [("-1", -1), ("6.0", 6.0), ("-5", -5), ("99", 99)]
facts = {"tree": str(TREE), "tag": TAG, "rows": {}}
for label, gidx in CASES:
    v, d, pos, home, moved = run(gidx)
    np.save(OUT / f"{TAG}_{label}_home.npy", home)
    np.save(OUT / f"{TAG}_{label}_after.npy", moved)
    facts["rows"][label] = {"verdict": v, "detail": d, "hand_xyz": [round(float(x), 4) for x in pos]}
    print(f"  gripper_dim_index={label:5s} -> {v:12s} hand={np.round(pos,4).tolist()}  {d[:80]}")

(OUT / f"facts_{TAG}.json").write_text(json.dumps(facts, indent=2))
print("wrote", OUT / f"facts_{TAG}.json")
