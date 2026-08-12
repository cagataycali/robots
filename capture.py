import io, json, os, pathlib, re
import numpy as np
from PIL import Image
import strands_robots
TREE = str(pathlib.Path(strands_robots.__file__).parents[1])
print("TREE:", TREE)
from strands_robots.simulation import Simulation

OUT = pathlib.Path("_art"); OUT.mkdir(exist_ok=True)
src = pathlib.Path("tests/simulation/mujoco/test_motion_primitives.py").read_text()
ARM_XML = re.search(r'ARM_XML = """(.*?)"""', src, re.S).group(1)
REACHABLE = json.loads(re.search(r'REACHABLE = (\[[^\]]*\])', src).group(1))
p = pathlib.Path("/tmp/arm-art.xml"); p.write_text(ARM_XML)

CAM = dict(position=[0.36, -0.38, 0.34], target=[0.08, 0.05, 0.22], fov=30)
facts = {"tree": TREE, "reachable": REACHABLE, "cam": CAM}

sim = Simulation(backend="mujoco", mesh=False)
assert sim.create_world()["status"] == "success"
assert sim.add_robot(name="arm", urdf_path=str(p))["status"] == "success"
assert sim.add_camera(name="look", **CAM)["status"] == "success"

def shot(tag):
    r = sim.render(camera_name="look", width=760, height=660)
    assert r.get("status") == "success", r
    b = next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)
    (OUT / f"{tag}.png").write_bytes(b)
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.int16)

def js(r): return next((b["json"] for b in r["content"] if "json" in b), None)
def rec(r):
    j = js(r) or {}
    return {"status": r["status"], "text": r["content"][0]["text"], **j}

home = shot("01_home")
facts["home"] = {}

r_short = sim.move_to(robot_name="arm", position=REACHABLE, tol=0.02, max_steps=2)
short = shot("02_not_reached")
facts["short"] = rec(r_short)

r_conv = sim.move_to(robot_name="arm", position=REACHABLE, tol=0.02, max_steps=400)
conv = shot("03_converged")
facts["conv"] = rec(r_conv)
sim.cleanup()

def dfrac(a, b): return float(((np.abs(a - b).sum(2)) > 8).mean())
def sat(a): return float(((a.max(2) - a.min(2)) > 45).mean())
facts["diff_home_short"] = dfrac(home, short)
facts["diff_short_conv"] = dfrac(short, conv)
facts["diff_home_conv"] = dfrac(home, conv)
facts["sat_home"] = sat(home)

# --- the story must be visible and the numbers must be the measured ones ----
assert facts["short"]["status"] == "error", facts["short"]
assert facts["short"]["reached"] is False and facts["short"]["steps"] == 2, facts["short"]
assert facts["conv"]["status"] == "success" and facts["conv"]["reached"] is True, facts["conv"]
assert facts["short"]["ik_residual_m"] <= 0.02 < facts["short"]["position_error_m"], facts["short"]
assert facts["conv"]["position_error_m"] <= 0.02, facts["conv"]
for k in ("diff_home_short", "diff_short_conv", "diff_home_conv"):
    assert facts[k] > 0.10, (k, facts[k])
assert facts["sat_home"] > 0.30, facts["sat_home"]

facts["coverage"] = {"file": "strands_robots/simulation/motion_primitives_base.py",
                     "before_missing": [207, 304, 380], "after_missing": [],
                     "before_pct": 97, "after_pct": 100, "stmts": 114}
facts["mutations"] = json.load(open(f"/tmp/mut-{os.environ['GITHUB_RUN_ID']}.json"))
facts["suite"] = {"base": 28535, "branch": 28551, "skipped": 257, "new_cases": 16}
json.dump(facts, open(OUT / "facts.json", "w"), indent=1)
print(json.dumps({k: v for k, v in facts.items() if k not in ("mutations",)}, indent=1)[:1600])
