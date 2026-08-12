"""Measure what a wrong-domain de-normalization does to a real arm.

Runs in whichever tree it is copied into (main or the branch) and dumps JSON.
"""
from __future__ import annotations
import json, pathlib, sys
import numpy as np

import strands_robots.policies.cosmos3.sim_ik as S
TREE = str(pathlib.Path(S.__file__).parents[3])
print("TREE:", TREE)

import mujoco
from robot_descriptions import panda_mj_description
from strands_robots.policies.cosmos3 import MinkIKBridge, decode_cosmos_chunk_to_targets
from strands_robots.policies.cosmos3.action_decode import load_action_stats
from strands_robots.policies.cosmos3.embodiments import get_embodiment

OUT = pathlib.Path(sys.argv[1])
emb = get_embodiment("umi")          # Cosmos3-Edge's forward-dynamics domain
D = emb.raw_action_dim

# The menagerie's scene.xml wraps the same panda with a floor, lights and a
# usable offscreen framebuffer; panda.xml alone renders unlit.
_SCENE = pathlib.Path(panda_mj_description.MJCF_PATH).parent / "scene.xml"
model = mujoco.MjModel.from_xml_path(str(_SCENE))
bridge = MinkIKBridge(model, ee_frame_name="hand", ee_frame_type="body")
q0 = np.zeros(model.nq); q0[:7] = [0, -0.3, 0, -2.2, 0, 2.0, 0.79]

# A steady forward+down Cosmos-style normalized chunk (umi: 16 steps, 10 cols).
chunk = np.zeros((emb.action_chunk_size, D), dtype=np.float32)
chunk[:, 0] = 0.55      # +x
chunk[:, 2] = -0.45     # -z
chunk[:, 3] = 1.0       # rot6d identity columns
chunk[:, 7] = 1.0
chunk[:, -1] = 0.2      # grasp

# umi ships no bundled quantiles; these stand in for "this domain's own".
own = {"q01": np.full(D, -0.02, np.float32), "q99": np.full(D, 0.02, np.float32)}
wrong_name = "bridge_orig_lerobot"
wrong = load_action_stats(wrong_name)

facts: dict = {"tree": TREE, "domain": emb.domain_name, "chunk": list(chunk.shape),
               "wrong_domain": wrong_name, "rows": {}}

import inspect
SUPPORTS = "stats_domain" in inspect.signature(decode_cosmos_chunk_to_targets).parameters
facts["supports_stats_domain"] = SUPPORTS


def run(label: str, **kw):
    # On the base tree the keyword does not exist; drop it so each tree runs the
    # same three logical rows with its own capability.
    if not SUPPORTS:
        kw.pop("stats_domain", None)
    try:
        out = decode_cosmos_chunk_to_targets(chunk, emb, bridge, q0, **kw)
        poses = np.asarray(out["poses"])
        travel = float(np.linalg.norm(poses[-1][:3, 3] - poses[0][:3, 3]))
        rec = {"outcome": "decoded", "qpos": out["qpos"].tolist(),
               "end_xyz": poses[-1][:3, 3].tolist(), "travel_m": travel}
    except Exception as e:
        rec = {"outcome": f"{type(e).__name__}", "message": str(e)}
    facts["rows"][label] = rec
    OUT.write_text(json.dumps(facts, indent=2))
    print(f"  {label:22s} -> {rec['outcome']}"
          + (f"  travel={rec['travel_m']:.4f} m" if "travel_m" in rec else ""))
    return rec

def render(qpos_last, name: str) -> str:
    """Render the arm at the final commanded joint target."""
    import imageio.v3 as iio
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    n = min(len(qpos_last), 7)
    data.qpos[:n] = np.asarray(qpos_last)[:n]
    mujoco.mj_forward(model, data)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    # Framing chosen by sweep (_art/sweep.py): maximises the visible difference
    # between the two decodes while keeping the arm well lit.
    cam.distance, cam.azimuth, cam.elevation = 1.05, 118.0, -12.0
    cam.lookat[:] = (0.42, 0.0, 0.45)
    with mujoco.Renderer(model, 460, 620) as r:
        r.update_scene(data, cam)
        img = r.render()
    path = f"/tmp/frame-{name}-{OUT.stem}.png"
    iio.imwrite(path, img)
    sat = float(((img.max(2).astype(int) - img.min(2)) > 45).mean())
    print(f"    rendered {name}: saturated={sat:.3f} -> {path}")
    return path


print("decoding umi chunk three ways:")
run("own_domain", stats=own, stats_domain=emb.domain_name)   # refused on main (no kwarg)
run("wrong_domain_declared", stats=wrong, stats_domain=wrong_name)
run("wrong_domain_bare", stats=wrong)                        # the silent path on main
# Render every row that produced joint targets, plus the untouched home pose.
facts["rows"]["home"] = {"outcome": "home", "qpos": q0[:7].tolist()}
for label, rec in list(facts["rows"].items()):
    if "qpos" in rec:
        last = rec["qpos"][-1] if isinstance(rec["qpos"][0], list) else rec["qpos"]
        rec["frame"] = render(last, label)
OUT.write_text(json.dumps(facts, indent=2))
print("wrote", OUT)
