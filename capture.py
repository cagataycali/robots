"""Measure the mesh example's teardown on one tree. Run once per tree."""
import json, os, pathlib, subprocess, sys, threading, time

import strands_robots
TREE = str(pathlib.Path(strands_robots.__file__).parents[1])
print("TREE:", TREE)
TAG = sys.argv[1]
OUT = pathlib.Path(f"/tmp/art-{TAG}-{os.environ['GITHUB_RUN_ID']}.json")
facts = {"tree": TREE, "tag": TAG}
def save():
    OUT.write_text(json.dumps(facts, indent=2))

# --- A: run the example exactly as a user does -------------------------------
env = dict(os.environ, PYTHONPATH=TREE, STRANDS_MESH_NAMESPACE=f"art-{TAG}")
t0 = time.time()
try:
    p = subprocess.run([sys.executable, "examples/04_mesh_peer_discovery.py"],
                       cwd=TREE, env=env, capture_output=True, text=True, timeout=30)
    rc, timed_out = p.returncode, False
except subprocess.TimeoutExpired:
    rc, timed_out = None, True
facts["script"] = {"exit": rc, "timed_out": timed_out, "wall_s": round(time.time() - t0, 1)}
save()
print("A:", facts["script"])

# --- B: what the cleanup line actually releases, in-process ------------------
os.environ.setdefault("STRANDS_MESH_LOCAL_DEV", "1")
os.environ["STRANDS_MESH_NAMESPACE"] = f"artp-{TAG}"
os.environ["MUJOCO_GL"] = "egl"
from strands_robots import Robot

src = pathlib.Path(TREE, "examples/04_mesh_peer_discovery.py").read_text()
read_name = "_mesh" if 'getattr(sim, "_mesh"' in src else "mesh"
sim = Robot("so100", mode="sim", mesh=True, peer_id="art-arm")
facts["attrs"] = {
    "factory_sets_mesh": getattr(sim, "mesh", None) is not None,
    "example_reads": read_name,
    "read_returns_none": getattr(sim, read_name, None) is None,
}
# Replicate the example's cleanup verbatim, then count what survives.
m = getattr(sim, read_name, None)
if m:
    m.stop()
time.sleep(1.5)
live = [t.name for t in threading.enumerate() if not t.daemon and t is not threading.main_thread()]
facts["threads"] = {"surviving_non_daemon": len(live), "names": sorted(set(live))[:4]}
save()
print("B:", facts["attrs"], facts["threads"])

# --- C: the simulation the example builds (must be identical across trees) ---
sim2 = Robot("so100", mode="sim", mesh=False)
sim2.add_camera(name="look", position=[0.62, -0.52, 0.42], target=[0.0, 0.0, 0.16], fov=42)
r = sim2.render(camera_name="look", width=560, height=460)
assert r.get("status") == "success", r
png = next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)
pathlib.Path(f"/tmp/art-{TAG}-{os.environ['GITHUB_RUN_ID']}.png").write_bytes(png)
facts["render"] = {"bytes": len(png)}
save()
# release the second world's session too (there is none: mesh=False)
mm = getattr(sim, "mesh", None)
if mm:
    mm.stop()
print("C: render", len(png), "bytes")
print("DONE", OUT)
