"""Record a real dataset through DatasetRecorder.create at a non-default camera
size, once with camera_dims keyed correctly and once with the key mistyped.
Dumps JSON + the frame decoded back out of each dataset's own MP4."""
import io, json, math, shutil, sys
from pathlib import Path
import numpy as np
import imageio.v3 as iio
import strands_robots
from strands_robots import dataset_recorder as rec_mod
from strands_robots.simulation import Simulation

TREE = str(Path(strands_robots.__file__).parents[1])
print("TREE:", TREE)
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
H, W, FPS, STEPS = 240, 320, 30, 12

ARM = """<mujoco model="probe">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81"/>
  <visual><headlight ambient="0.6 0.6 0.6" diffuse="0.7 0.7 0.7"/></visual>
  <asset><texture type="skybox" builtin="gradient" rgb1="0.35 0.5 0.75" rgb2="0.1 0.15 0.3" width="64" height="64"/></asset>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 .1" rgba=".55 .55 .6 1"/>
    <body name="base" pos="0 0 0.05">
      <geom type="cylinder" size="0.05 0.05" rgba=".3 .3 .35 1"/>
      <body name="link1" pos="0 0 0.06">
        <joint name="shoulder" type="hinge" axis="0 1 0" range="-1.4 1.4" damping="1.2"/>
        <geom type="capsule" fromto="0 0 0 0.26 0 0" size="0.028" rgba=".95 .55 .12 1"/>
        <body name="link2" pos="0.26 0 0">
          <joint name="elbow" type="hinge" axis="0 1 0" range="-1.4 1.4" damping="0.9"/>
          <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.022" rgba=".25 .75 .35 1"/>
          <body name="tip" pos="0.2 0 0"><geom type="sphere" size="0.035" rgba=".9 .2 .25 1"/></body>
        </body>
      </body>
    </body>
    <camera name="look" pos="0.62 -0.6 0.42" mode="targetbody" target="base" fovy="40"/>
  </worldbody>
  <actuator>
    <position name="a_shoulder" joint="shoulder" kp="22" ctrlrange="-1.4 1.4"/>
    <position name="a_elbow" joint="elbow" kp="18" ctrlrange="-1.4 1.4"/>
  </actuator>
</mujoco>
"""

def render(sim):
    # A camera declared in an MJCF attached through add_robot is namespaced to
    # "<robot>/<camera>"; the bare name does not resolve.
    r = sim.render(camera_name="arm/look", width=W, height=H)
    if r.get("status") != "success":
        raise RuntimeError(f"render failed: {r}")
    png = next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)
    return np.asarray(iio.imread(io.BytesIO(png)))[:, :, :3]

def render_frames():
    """Render the rollout once, headless, and hand the frames back as arrays.

    The sim is torn down before any dataset is created: an EGL context and
    lerobot's video encoder in one process deadlock, and the frames are the
    only thing the recorder needs.
    """
    xml = OUT / "arm.xml"; xml.write_text(ARM)
    sim = Simulation(backend="mujoco", mesh=False)
    sim.create_world()
    sim.add_robot(name="arm", urdf_path=str(xml))
    frames, actions = [], []
    for i in range(STEPS):
        f = (i + 1) / STEPS
        act = {"a_shoulder": -0.35 - 0.75 * f, "a_elbow": 0.25 + 0.9 * f}
        sim.send_action(act, robot_name="arm", n_substeps=10)
        frames.append(render(sim)); actions.append(act)
    sim.cleanup()
    return frames, actions


def scenario(name, camera_dims, frames, actions):
    root = OUT / f"ds_{name}"
    shutil.rmtree(root, ignore_errors=True)
    res = {"scenario": name, "camera_dims": repr(camera_dims)}
    try:
        rec = rec_mod.DatasetRecorder.create(
            repo_id=f"local/{name}", fps=FPS, robot_type="probe",
            joint_names=["shoulder", "elbow"], camera_keys=["image"],
            camera_dims=camera_dims, video_width=640, video_height=480,
            task="reach", root=str(root),
        )
    except Exception as e:
        res.update(create="REFUSED", message=f"{type(e).__name__}: {e}",
                   declared=None, frames=0, mp4=None, decoded_frames=0)
        return res, None
    res["create"] = "success"
    res["declared"] = str(rec.dataset.meta.info["features"]["observation.images.image"]["shape"])
    err = None
    for img, act in zip(frames, actions, strict=True):
        obs = {"shoulder": act["a_shoulder"], "elbow": act["a_elbow"], "image": img}
        try:
            rec.add_frame(obs, {"shoulder": act["a_shoulder"], "elbow": act["a_elbow"]}, task="reach")
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:110]}"; break
    if err is None:
        rec.save_episode(); rec.finalize()
    res["add_frame_error"] = err
    res["frames"] = int(getattr(rec.dataset.meta, "total_frames", 0) or 0)
    mp4s = sorted(root.rglob("*.mp4"))
    res["mp4"] = str(mp4s[0].relative_to(root)) if mp4s else None
    out = None
    res["decoded_frames"] = 0
    if mp4s:
        decoded = list(iio.imiter(mp4s[0]))
        res["decoded_frames"] = len(decoded)
        if decoded:
            out = np.asarray(decoded[len(decoded) // 2])
    return res, out


# Both trees must encode the SAME source frames: the recorder is the variable
# under test, and two independent MuJoCo renders differ by renderer noise that
# lossy H.264 then amplifies. The first run renders and caches; the second loads.
CACHE = Path("/tmp/artv2/frames")
if (CACHE / "act.json").exists():
    FR = [np.load(CACHE / f"f{i:03d}.npy") for i in range(STEPS)]
    ACT = json.loads((CACHE / "act.json").read_text())
    print(f"loaded {len(FR)} cached frames of {FR[0].shape}", flush=True)
else:
    FR, ACT = render_frames()
    CACHE.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(FR):
        np.save(CACHE / f"f{i:03d}.npy", f)
    (CACHE / "act.json").write_text(json.dumps(ACT))
    print(f"rendered {len(FR)} frames of {FR[0].shape}", flush=True)
np.save(OUT / "source_frame.npy", FR[len(FR) // 2])
rows = []
for name, dims in [("honored", {"image": (H, W)}), ("typo", {"imagee": (H, W)})]:
    r, f = scenario(name, dims, FR, ACT)
    rows.append(r)
    if f is not None:
        np.save(OUT / f"frame_{name}.npy", f)
    print(name, "->", json.dumps(r))
(OUT / "facts.json").write_text(json.dumps({"tree": TREE, "rows": rows}, indent=2))
print("WROTE", OUT / "facts.json")
