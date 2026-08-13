import json, os, pathlib, sys
import numpy as np, imageio.v3 as iio
r = os.environ["GITHUB_RUN_ID"]
F = json.loads(pathlib.Path(f"/tmp/facts-robots-mine-{r}.json").read_text())
sys.path.insert(0, str(pathlib.Path.cwd()))
from strands_robots import Simulation
XML = pathlib.Path(f"/tmp/arm5-robots-mine-{r}.xml").read_text()
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_roll", "jaw"]
poses = {"rollout": F["rollout_pose"], "primitive": F["primitive_pose"]}
CANDS = [
    ("A", [0.34, -0.36, 0.40], [0.02, 0.02, 0.20], 34),
    ("B", [0.22, -0.24, 0.34], [0.00, 0.02, 0.26], 40),
    ("C", [0.17, -0.19, 0.32], [0.00, 0.01, 0.29], 44),
    ("D", [0.26, -0.10, 0.38], [0.01, 0.01, 0.27], 36),
    ("E", [0.13, -0.15, 0.34], [0.00, 0.00, 0.31], 50),
]
p = pathlib.Path(f"/tmp/arm5-sweep-{r}.xml"); p.write_text(XML)
for cid, pos, tgt, fov in CANDS:
    imgs = {}
    for lab, q in poses.items():
        sim = Simulation(backend="mujoco", mesh=False)
        try:
            sim.create_world(ground_plane=False)
            sim.add_robot(name="arm", urdf_path=str(p))
            sim.add_camera(name="c", position=pos, target=tgt, fov=fov)
            sim.set_joint_positions(dict(zip(JOINTS, q, strict=True)))
            res = sim.render(camera_name="c", width=720, height=660)
            assert res.get("status") == "success", res
            b = next(c["image"]["source"]["bytes"] for c in res["content"] if "image" in c)
            fp = pathlib.Path(f"/tmp/sw-{cid}-{lab}-{r}.png"); fp.write_bytes(b)
            imgs[lab] = np.asarray(iio.imread(fp)).astype(int)
        finally:
            sim.cleanup()
    a, bb = imgs["rollout"], imgs["primitive"]
    diff = (np.abs(a - bb).max(2) > 8)
    sat = min(float(((im.max(2) - im.min(2)) > 45).mean()) for im in imgs.values())
    ys, xs = np.nonzero(diff)
    crop = 0.0
    if len(ys):
        pad = 30
        y0, y1 = max(0, ys.min()-pad), min(diff.shape[0], ys.max()+pad)
        x0, x1 = max(0, xs.min()-pad), min(diff.shape[1], xs.max()+pad)
        crop = float(diff[y0:y1, x0:x1].mean())
    print(f"{cid} fov={fov:>2} full={100*diff.mean():5.2f}%  cropped={100*crop:5.2f}%  min_sat={sat:.3f}")
