import json, os, pathlib
import numpy as np, mujoco
from robot_descriptions import panda_mj_description

RUN = os.environ["GITHUB_RUN_ID"]
A = json.loads(pathlib.Path(f"/tmp/art-main-{RUN}.json").read_text())
scene = pathlib.Path(panda_mj_description.MJCF_PATH).parent / "scene.xml"
model = mujoco.MjModel.from_xml_path(str(scene))

def pose(label):
    q = A["rows"][label]["qpos"][-1]
    d = mujoco.MjData(model); d.qpos[:] = 0.0
    d.qpos[:min(len(q), 7)] = np.asarray(q)[:7]
    mujoco.mj_forward(model, d); return d

pa, pb = pose("own_domain"), pose("wrong_domain_bare")

def shot(data, cam, r):
    r.update_scene(data, cam); return r.render()[:, :, :3].astype(int)

cands = [
    (1.55, 132, -18, (0.35, 0.0, 0.35)),
    (1.15, 132, -14, (0.40, 0.0, 0.40)),
    (0.95, 140, -10, (0.45, 0.0, 0.42)),
    (1.05, 118, -12, (0.42, 0.0, 0.45)),
    (0.85, 150, -8,  (0.48, 0.0, 0.40)),
]
with mujoco.Renderer(model, 460, 620) as r:
    for dist, az, el, look in cands:
        cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(model, cam)
        cam.distance, cam.azimuth, cam.elevation = dist, az, el
        cam.lookat[:] = look
        ia, ib = shot(pa, cam, r), shot(pb, cam, r)
        frac = float((np.abs(ia - ib).sum(2) > 24).mean())
        sat = float(((ia.max(2) - ia.min(2)) > 45).mean())
        print(f"dist={dist:<5} az={az:<4} el={el:<4} look={look}  diff={frac:6.2%}  sat={sat:.3f}")
