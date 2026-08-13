"""Probe Isaac start_recording camera scoping: raw / schema-safe / unknown / dedup + hook mapping."""
import json, os, pathlib, sys, threading
import numpy as np
import strands_robots
print("TREE:", pathlib.Path(strands_robots.__file__).parents[1])

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from strands_robots.simulation.isaac.config import IsaacConfig
from strands_robots.simulation.isaac.simulation import IsaacSimulation, _CameraState, _RobotState

JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]

class _Handle:
    def __init__(self, rgba): self.rgba = rgba
    def get_rgba(self): return self.rgba

def _cam(name, w=64, h=48, fill=0):
    c = _CameraState(name=name, prim_path=f"/World/Cameras/{name}", width=w, height=h)
    c.handle = _Handle(np.full((h, w, 4), fill, np.uint8))
    return c

def _engine(cams):
    e = IsaacSimulation.__new__(IsaacSimulation)
    e._config = IsaacConfig(render_mode="rtx_realtime")
    e._lock = threading.RLock(); e._world = None; e._world_created = True
    e._robots = {"so100": _RobotState(name="so100", prim_path="/World/Robots/so100",
                                      joint_names=list(JOINTS), data_config="so100")}
    e._cameras = cams; e._objects = {}; e._prim_registry = []
    e._cams_rec_state = None; e._recording_state_dict = {}; e._action_controllers = {}
    e._sim_time = 0.0; e._step_count = 0; e._replicated = False
    e._num_envs_active = 1; e._pump_running = False; e._main_tid = threading.get_ident()
    return e

TMP = pathlib.Path(sys.argv[1]); TMP.mkdir(parents=True, exist_ok=True)
rows = []

def run(label, cams_arg, cam_dict=None):
    cams = cam_dict if cam_dict is not None else {
        "arm0/wrist": _cam("arm0/wrist", fill=11), "overview": _cam("overview", fill=22)}
    e = _engine(cams)
    root = str(TMP / label)
    r = e.start_recording(repo_id=f"local/probe_{label}", root=root, fps=30, overwrite=True, cameras=cams_arg)
    row = {"label": label, "cameras_arg": cams_arg, "status": r.get("status")}
    if r.get("status") == "success":
        rec = e._recording_state_dict["dataset_recorder"]
        row["image_feats"] = sorted(k for k in rec.dataset.features if k.startswith("observation.images."))
        row["recording_cameras"] = [(t[0], t[1]) for t in e._recording_state_dict["recording_cameras"]]
        # drive one frame through the REAL hook and see which columns get data
        hook = e._make_run_policy_hook("so100", "probe")
        obs = {j: 0.1 for j in JOINTS}
        for cn, c in cams.items():
            obs[cn] = np.asarray(c.handle.get_rgba())[..., :3].astype(np.uint8)
        hook(0, obs, {j: 0.2 for j in JOINTS})
        row["frames_after_one_hook"] = int(rec.frame_count)
        e.stop_recording()
    else:
        row["text"] = r["content"][0]["text"][:160]
    rows.append(row)
    print(json.dumps(row, indent=2))

run("raw", ["arm0/wrist"])
run("safe", ["arm0__wrist"])
run("unknown", ["nope"])
run("both_spellings", ["arm0/wrist", "arm0__wrist"])
run("all_default", None)
(TMP / "rows.json").write_text(json.dumps(rows, indent=2))
