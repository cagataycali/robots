import pathlib, re, json, numpy as np
import strands_robots
print("TREE:", pathlib.Path(strands_robots.__file__).parents[1])
from strands_robots.simulation import Simulation

src = pathlib.Path("tests/simulation/mujoco/test_motion_primitives.py").read_text()
ARM_XML = re.search(r'ARM_XML = """(.*?)"""', src, re.S).group(1)
REACHABLE = json.loads(re.search(r'REACHABLE = (\[[^\]]*\])', src).group(1))
p = pathlib.Path("/tmp/arm-art.xml"); p.write_text(ARM_XML)

def png(sim, cam, w=760, h=660):
    r = sim.render(camera_name=cam, width=w, height=h)
    assert r.get("status") == "success", r
    return next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)

import io
from PIL import Image
def arr(b): return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.int16)
def diff(a, b): return float(((np.abs(a - b).sum(2)) > 8).mean())
def sat(a): return float(((a.max(2) - a.min(2)) > 45).mean())

CAMS = [
    ("A", [0.45, -0.45, 0.38], [0.10, 0.05, 0.16], 40),
    ("B", [0.36, -0.38, 0.34], [0.08, 0.05, 0.22], 30),
    ("C", [0.40, -0.34, 0.30], [0.12, 0.06, 0.18], 34),
    ("D", [0.30, -0.42, 0.30], [0.10, 0.08, 0.18], 32),
]
for name, pos, tgt, fov in CAMS:
    sim = Simulation(backend="mujoco", mesh=False)
    sim.create_world(); sim.add_robot(name="arm", urdf_path=str(p))
    sim.add_camera(name=f"c{name}", position=pos, target=tgt, fov=fov)
    home = arr(png(sim, f"c{name}"))
    sim.move_to(robot_name="arm", position=REACHABLE, tol=0.02, max_steps=2)
    short = arr(png(sim, f"c{name}"))
    sim.move_to(robot_name="arm", position=REACHABLE, tol=0.02, max_steps=400)
    conv = arr(png(sim, f"c{name}"))
    print(f"  cam {name} fov={fov}: sat={sat(home):.3f}  home-vs-2step={diff(home,short)*100:5.2f}%  "
          f"home-vs-converged={diff(home,conv)*100:5.2f}%  2step-vs-converged={diff(short,conv)*100:5.2f}%")
    sim.cleanup()
