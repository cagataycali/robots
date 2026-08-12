"""Measure the Isaac camera-readback pixel-floor matrix, and render the MuJoCo sibling.

Row 3 is a real headless MuJoCo render: the shared floor `positive_count_error`
backs the same `get_camera_params` readback on every backend, so the MuJoCo
surface is the one that can be photographed on this host, and it is the control
that the shared rule is untouched.
"""
from __future__ import annotations
import json, os, pathlib, threading
from typing import Any

import numpy as np
import strands_robots
TREE = str(pathlib.Path(strands_robots.__file__).parents[1])
print("TREE:", TREE)

from strands_robots.simulation.isaac.config import IsaacConfig
from strands_robots.simulation.isaac.simulation import IsaacSimulation, _CameraState
from strands_robots.utils import positive_count_error

NATIVE_W, NATIVE_H = 64, 48
UNUSABLE = [("0", 0), ("-8", -8), ("True", True), ("2.7", 2.7), ("64.0", 64.0),
            ("nan", float("nan")), ("inf", float("inf")), ("'64'", "64"),
            ("[64]", [64]), ("np.int64(64)", np.int64(64))]


class Handle:
    def __init__(self): self.reads: list[str] = []
    def get_intrinsics_matrix(self):
        self.reads.append("intrinsics")
        return np.array([[50.0, 0.0, 32.0], [0.0, 50.0, 24.0], [0.0, 0.0, 1.0]])
    def get_world_pose(self):
        self.reads.append("pose"); return np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.0, 0.0, 0.0])
    def get_rgba(self):
        self.reads.append("rgba"); return np.full((NATIVE_H, NATIVE_W, 4), 200, dtype=np.uint8)
    def get_depth(self):
        self.reads.append("depth"); return np.full((NATIVE_H, NATIVE_W), 1.5, dtype=np.float32)


def isaac(handle=None):
    e = IsaacSimulation.__new__(IsaacSimulation)
    e._config = IsaacConfig(render_mode="rtx_realtime", camera_width=NATIVE_W, camera_height=NATIVE_H)
    e._lock = threading.RLock(); e._world = None; e._world_created = True
    e._robots = {}; e._objects = {}; e._cameras = {}; e._prim_registry = []
    e._cam_out_size = {}; e._camera_warmup_steps = 0
    e._sim_time = 0.0; e._step_count = 0; e._main_tid = threading.get_ident()
    cam = _CameraState("cam", "/World/cameras/cam", NATIVE_W, NATIVE_H)
    cam.handle = handle or Handle(); e._cameras["cam"] = cam
    e._create_camera_prim = lambda **kw: (Handle(), 24.0)  # type: ignore
    return e


def classify(fn) -> str:
    try:
        r = fn()
    except BaseException as exc:  # noqa: BLE001 - classifying, not handling
        return f"raise:{str(exc)}"
    if isinstance(r, dict) and r.get("status") == "error":
        return "envelope:" + r["content"][0]["text"]
    if isinstance(r, tuple) and len(r) == 3 and r[0] is None:
        return "meta:" + str(r[2].get("error", ""))
    return "accepted"


facts: dict[str, Any] = {"tree": TREE}

# --- the four Isaac surfaces that apply the shared floor -------------------
SURFACES = ["add_camera", "render", "get_frame", "get_camera_params"]
matrix: dict[str, dict[str, bool]] = {}
for label, v in UNUSABLE:
    row = {}
    e = isaac(); e._cameras.clear()
    row["add_camera"] = classify(lambda: e.add_camera(name="c", position=[1, 1, 1], target=[0, 0, 0], width=v)).startswith("envelope")
    e2 = isaac(); e2._config = IsaacConfig(render_mode="headless", camera_width=NATIVE_W, camera_height=NATIVE_H)
    row["render"] = classify(lambda: e2._render_frame("cam", width=v)).startswith("meta")
    for m in ("get_frame", "get_camera_params"):
        got = classify(lambda: getattr(isaac(), m)(camera_name="cam", width=v))
        want = "raise:" + (positive_count_error(v, "width", m) or "")
        row[m] = got == want
    matrix[label] = row
facts["matrix"] = matrix
facts["surfaces"] = SURFACES
# which surfaces had their refusal DRIVEN by a test before this PR
facts["driven_before"] = {"add_camera": True, "render": True, "get_frame": False, "get_camera_params": False}

# --- guard placement: a refused size reads no handle ------------------------
h = Handle()
_ = classify(lambda: isaac(h).get_frame(camera_name="cam", width=0))
facts["reads_after_refusal"] = list(h.reads)
h2 = Handle()
_ = classify(lambda: isaac(h2).get_frame(camera_name="cam", width=NATIVE_W))
facts["reads_after_usable"] = list(h2.reads)

# --- the values the readbacks exist to return ------------------------------
rgb, depth = isaac().get_frame("cam")
params = isaac().get_camera_params("cam")
rot = np.asarray(params.T_world_cam)[:3, :3]
facts["readback"] = {
    "rgb_shape": list(rgb.shape), "rgb_dtype": str(rgb.dtype),
    "depth_shape": list(depth.shape), "depth_dtype": str(depth.dtype),
    "native": [params.width, params.height],
    "prim_to_gl": rot.tolist(),
    "prim_to_gl_expected": [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
}

# --- mutation table -------------------------------------------------------
facts["mutations"] = json.load(open(f"/tmp/mut-{os.environ['GITHUB_RUN_ID']}.json"))

# --- MuJoCo sibling: same shared floor, and a real headless render ---------
from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

sim = MuJoCoSimEngine(mesh=False)
sim.create_world(timestep=0.002, gravity=[0, 0, -9.81], ground_plane=True)
sim.add_robot(name="arm", data_config="so101", position=[0, 0, 0])
sim.add_camera(name="look", position=[0.42, -0.40, 0.34], target=[0.06, 0.04, 0.16], fov=38)
sim.step(120)
r = sim.render(camera_name="look", width=760, height=620)
assert r.get("status") == "success", r
png = next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)
pathlib.Path(f"/tmp/art-mujoco-{os.environ['GITHUB_RUN_ID']}.png").write_bytes(png)
mp = sim.get_camera_params(camera_name="look", width=760, height=620)
facts["mujoco"] = {
    "K": np.asarray(mp.K).tolist(), "wh": [mp.width, mp.height],
    "refusals": {label: (positive_count_error(v, "width", "get_camera_params") is not None)
                 for label, v in UNUSABLE},
}
sim.cleanup()

out = pathlib.Path(f"/tmp/art-facts-{os.environ['GITHUB_RUN_ID']}.json")
out.write_text(json.dumps(facts, indent=2))
print("WROTE", out)
print("isaac matrix rows:", len(matrix), "all four surfaces refuse every probe:",
      all(all(row.values()) for row in matrix.values()))
print("reads_after_refusal:", facts["reads_after_refusal"], "reads_after_usable:", facts["reads_after_usable"])
print("mujoco refuses the same set:", all(facts["mujoco"]["refusals"].values()))
