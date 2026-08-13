"""Measure the finalizer's report and the teardown it reaches, in a given tree."""
import json, os, pathlib, subprocess, sys
import numpy as np
import strands_robots
TREE = str(pathlib.Path(strands_robots.__file__).parents[1])
OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
TAG = sys.argv[2]
facts = {"tree": TREE, "tag": TAG}
def save(): (OUT / f"facts-{TAG}.json").write_text(json.dumps(facts, indent=2))

ROBOTS = ["so101", "unitree_g1", "unitree_go2"]

# --- 1. the reporter's scenario, in a child process, both modes -------------
SCRIPT = '''
import sys
from strands_robots import Robot
mode = sys.argv[1]
sims = [Robot(n, mesh=False) for n in ["so101", "unitree_g1", "unitree_go2"]]
for s in sims:
    s.render(width=64, height=64); s.step(2)
if mode == "cleanup":
    for s in sims:
        s.cleanup()
sys.stderr.write("SCENARIO_OK\\n")
'''
sp = OUT / "scenario.py"; sp.write_text(SCRIPT)
for mode in ("nocleanup", "cleanup"):
    r = subprocess.run([sys.executable, str(sp), mode], capture_output=True, text=True,
                       timeout=600, env={**os.environ, "MUJOCO_GL": "egl", "PYTHONPATH": TREE})
    facts[f"scenario_{mode}"] = {
        "exit": r.returncode,
        "ok": "SCENARIO_OK" in r.stderr,
        "cleanup_warnings": r.stderr.count("Cleanup error during __del__"),
        "egl_errors": r.stderr.count("EGLError"),
    }
save()

# --- 2. which teardown steps the finalizer reaches at real shutdown --------
TRACE = '''
import os, sys
from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine
fd = os.open(sys.argv[1], os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
for name in ("_shutdown_ros_bridge", "_close_main_thread_renderers"):
    real = getattr(MuJoCoSimEngine, name)
    def w(self, *a, _r=real, _w=os.write, _fd=fd, _n=name.encode(), **k):
        _w(_fd, _n + b"\\n"); return _r(self, *a, **k)
    setattr(MuJoCoSimEngine, name, w)
real_c = MuJoCoSimEngine.cleanup
def ct(self, *a, _r=real_c, _w=os.write, _fd=fd, **k):
    _w(_fd, b"enter\\n")
    try: out = _r(self, *a, **k)
    except BaseException as e:
        _w(_fd, b"raised:" + type(e).__name__.encode() + b"\\n"); raise
    _w(_fd, b"returned\\n"); return out
MuJoCoSimEngine.cleanup = ct
engine = MuJoCoSimEngine(tool_name="probe")
sys.stderr.write("built\\n")
'''
tp = OUT / "trace.py"; tp.write_text(TRACE)
tf = OUT / f"trace-{TAG}.txt"
r = subprocess.run([sys.executable, str(tp), str(tf)], capture_output=True, text=True,
                   timeout=300, env={**os.environ, "PYTHONPATH": TREE})
facts["shutdown_trace"] = tf.read_text().split() if tf.exists() else []
facts["shutdown_warnings"] = r.stderr.count("Cleanup error during __del__")
save()

# --- 3. one real render per robot: the sim path must be untouched ----------
from strands_robots import Robot
cams = {"so101": ([0.62,-0.52,0.42],[0,0,0.16],42),
        "unitree_g1": ([2.1,-1.9,1.15],[0,0,0.62],34),
        "unitree_go2": ([1.6,-1.5,0.95],[0,0,0.30],36)}
for name in ROBOTS:
    s = Robot(name, mesh=False)
    pos, tgt, fov = cams[name]
    s.add_camera(name="look", position=pos, target=tgt, fov=fov)
    s.step(120)
    res = s.render(camera_name="look", width=420, height=380)
    assert res.get("status") == "success", res
    png = next(c["image"]["source"]["bytes"] for c in res["content"] if "image" in c)
    (OUT / f"{name}-{TAG}.png").write_bytes(png)
    import imageio.v3 as iio
    arr = iio.imread(png)
    np.save(OUT / f"{name}-{TAG}.npy", arr)
    sat = float((((arr.max(2).astype(int) - arr.min(2)) > 45) | (arr.mean(2) > 88)).mean())
    facts.setdefault("renders", {})[name] = {"shape": list(arr.shape), "content_frac": round(sat, 4)}
    assert sat > 0.15, (name, sat)
    s.cleanup()
save()
print("captured", TAG, "->", OUT)
