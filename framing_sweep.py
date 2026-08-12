"""Framing sweep: render the two measured base poses from one tree."""
import pathlib, numpy as np, strands_robots, mujoco as mj, imageio.v3 as iio
print("TREE:", pathlib.Path(strands_robots.__file__).parents[1], flush=True)
from strands_robots.simulation import Simulation

GOOD = [0.0, 0.8, 0.445, 1.0, 0.0, 0.0, 0.0]
BAD = [0.0, 0.0, 0.0, 0.0, -0.000966, -5.2e-05, -0.001201]
CANDS = [
    ("A", [1.55, -1.35, 0.95], [0.30, 0.40, 0.18], 45),
    ("B", [1.20, -0.95, 0.80], [0.22, 0.38, 0.18], 42),
    ("C", [1.00, -0.80, 0.70], [0.16, 0.36, 0.16], 40),
    ("D", [0.90, -0.75, 0.62], [0.10, 0.38, 0.14], 38),
    ("E", [1.10, -0.55, 0.72], [0.05, 0.40, 0.16], 40),
]

sim = Simulation(mesh=False); sim.create_world()
for i in range(2):
    cfg = "so101" if i % 2 == 0 else "so100"
    sim.add_robot(name=f"arm_{i}", data_config=cfg, position=[0.6 * i, 0.0, 0.0])
    keys = sim.robot_action_keys(f"arm_{i}")
    sim.send_action(dict(zip(keys, [0.7, -0.9, 1.1, 0.0, 0.0, 0.4][: len(keys)])), robot_name=f"arm_{i}")
    sim.step(300)
sim.add_robot(name="base", data_config="go2", position=[0.0, 0.8, 0.0], keyframe="home")
# Every camera FIRST: add_camera recompiles and rebinds _world._model/_data, so a
# handle captured before it is stale and writes land on an orphaned buffer.
for tag, pos, tgt, fov in CANDS:
    assert sim.add_camera(name=f"cam{tag}", position=pos, target=tgt, fov=fov)["status"] == "success"

def shot(name):
    r = sim.render(camera_name=name, width=900, height=700)
    assert r.get("status") == "success", r
    return np.asarray(iio.imread(next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)))

for tag, pos, tgt, fov in CANDS:
    m, d = sim._world._model, sim._world._data
    fj = [j for j in range(m.njnt) if int(m.jnt_type[j]) == int(mj.mjtJoint.mjJNT_FREE)][0]
    adr = int(m.jnt_qposadr[fj])
    d.qpos[adr:adr+7] = GOOD; mj.mj_forward(m, d); good = shot(f"cam{tag}").astype(int)
    d.qpos[adr:adr+7] = BAD;  mj.mj_forward(m, d); bad = shot(f"cam{tag}").astype(int)
    diff = np.abs(good - bad).sum(2)
    sat = ((good.max(2) - good.min(2)) > 45).mean()
    print(f"  {tag}: differing(>8)={(diff>8).mean()*100:5.2f}%  structure={sat*100:5.2f}%  fov={fov} pos={pos}", flush=True)
sim.destroy()
