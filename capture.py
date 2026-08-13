"""Measure the Isaac camera-scoping matrix and round-trip one aliased frame."""
import json, os, pathlib, sys, threading
import numpy as np
import strands_robots
ROOT = pathlib.Path(strands_robots.__file__).parents[1]
print("TREE:", ROOT)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from strands_robots.simulation.isaac.config import IsaacConfig
from strands_robots.simulation.isaac.simulation import IsaacSimulation, _CameraState, _RobotState

JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
W, H = 128, 96

def _pattern(step: int) -> np.ndarray:
    """Recognizable synthetic wrist frame: a moving bar over a gradient."""
    y, x = np.mgrid[0:H, 0:W]
    img = np.zeros((H, W, 3), np.uint8)
    img[..., 0] = (x * 255 // W).astype(np.uint8)
    img[..., 2] = (y * 255 // H).astype(np.uint8)
    col = (step * 23 + 12) % (W - 14)
    img[H // 4 : 3 * H // 4, col : col + 14] = (250, 230, 40)
    return img

class _Handle:
    def __init__(self): self.step = 0
    def get_rgba(self):
        rgb = _pattern(self.step); self.step += 1
        return np.dstack([rgb, np.full((H, W, 1), 255, np.uint8)])

def _cam(name):
    c = _CameraState(name=name, prim_path=f"/World/Cameras/{name}", width=W, height=H)
    c.handle = _Handle(); return c

def _engine():
    e = IsaacSimulation.__new__(IsaacSimulation)
    e._config = IsaacConfig(render_mode="rtx_realtime")
    e._lock = threading.RLock(); e._world = None; e._world_created = True
    e._robots = {"so100": _RobotState(name="so100", prim_path="/World/Robots/so100",
                                      joint_names=list(JOINTS), data_config="so100")}
    e._cameras = {"arm0/wrist": _cam("arm0/wrist"), "overview": _cam("overview")}
    e._objects = {}; e._prim_registry = []; e._cams_rec_state = None
    e._recording_state_dict = {}; e._action_controllers = {}
    e._sim_time = 0.0; e._step_count = 0; e._replicated = False
    e._num_envs_active = 1; e._pump_running = False; e._main_tid = threading.get_ident()
    return e

TMP = pathlib.Path(sys.argv[1]); TMP.mkdir(parents=True, exist_ok=True)
facts = {"tree": str(ROOT), "cells": [], "roundtrip": {}}

def record(label, cams, n_frames=0):
    e = _engine(); root = TMP / label
    r = e.start_recording(repo_id=f"local/art_{label}", root=str(root), fps=30,
                          overwrite=True, cameras=cams)
    row = {"label": label, "requested": cams, "status": r.get("status")}
    if r.get("status") != "success":
        row["text"] = r["content"][0]["text"][:200]; facts["cells"].append(row); return row, None
    rec = e._recording_state_dict["dataset_recorder"]
    row["columns"] = sorted(k.split(".")[-1] for k in rec.dataset.features
                            if k.startswith("observation.images."))
    row["sources"] = [t[0] for t in e._recording_state_dict["recording_cameras"]]
    if n_frames:
        hook = e._make_run_policy_hook("so100", "artifact")
        for i in range(n_frames):
            obs = {j: 0.05 * i for j in JOINTS}
            for cn, c in e._cameras.items():
                obs[cn] = np.asarray(c.handle.get_rgba())[..., :3]
            hook(i, obs, {j: 0.06 * i for j in JOINTS})
        row["frames"] = int(rec.frame_count)
        e.stop_recording()
    facts["cells"].append(row); return row, root

safe_row, safe_root = record("safe", ["arm0__wrist"], n_frames=4)
raw_row, _ = record("raw", ["arm0/wrist"])
record("both", ["arm0/wrist", "arm0__wrist"])
record("unknown", ["nope"])
record("all", None)

# The alias and raw spellings must be indistinguishable.
facts["equivalent"] = (safe_row["columns"] == raw_row["columns"]
                       and safe_row["sources"] == raw_row["sources"])

# Round-trip: decode the MP4 the ALIASED column produced.
import imageio.v3 as iio
mp4 = sorted(pathlib.Path(safe_root).rglob("*.mp4"))
assert mp4, f"no MP4 under {safe_root}"
frames = list(iio.imiter(mp4[0]))
arr = np.asarray(frames[2])
np.save(TMP / "decoded.npy", arr)
facts["roundtrip"] = {
    "mp4": str(mp4[0].relative_to(safe_root)), "decoded_frames": len(frames),
    "shape": list(arr.shape), "size_bytes": mp4[0].stat().st_size,
    "saturated_frac": round(float(((arr.max(2).astype(int) - arr.min(2)) > 45).mean()), 4),
}

# Mutation table, measured by _probe/mutate.py (published alongside).
facts["mutations"] = [
    ("M1 drop the schema-safe alias branch", 3, 0),
    ("M2 alias keeps the safe name as its render source", 2, 0),
    ("M3 drop the both-spellings dedup", 1, 0),
    ("M4 raw request does not canonicalize", 2, 0),
    ("M5 alias swaps raw/safe", 3, 0),
]
facts["coverage"] = {"before_missing": 12, "after_missing": 11,
                     "before_pct": 92.73, "after_pct": 93.33, "closed": [338]}
facts["cross_backend"] = [
    ("all cameras by default", 1, 1, 1, 1),
    ("subset by raw name", 1, 1, 1, 1),
    ("schema-safe alias", 1, 1, 0, 1),
    ("raw == safe equivalence", 1, 0, 0, 1),
    ("unknown fails loudly", 1, 1, 1, 1),
    ("both-spellings dedup", 0, 0, 0, 1),
]
assert facts["equivalent"], "raw and schema-safe must agree"
assert safe_row["frames"] == 4, safe_row
assert facts["roundtrip"]["saturated_frac"] > 0.10, facts["roundtrip"]
(TMP / "facts.json").write_text(json.dumps(facts, indent=2))
print(json.dumps({k: v for k, v in facts.items() if k != "cells"}, indent=2))
print("cells:", json.dumps(facts["cells"], indent=2))
