import json, pathlib, sys
import numpy as np
import strands_robots
print("TREE:", pathlib.Path(strands_robots.__file__).parents[1])
sys.path.insert(0, ".")
from tests.simulation.mujoco.test_motion_primitives import ARM_XML, REACHABLE
from strands_robots.simulation.mujoco.simulation import Simulation

OUT = pathlib.Path("/tmp/art"); OUT.mkdir(exist_ok=True)
xml = pathlib.Path("/tmp/art_arm.xml"); xml.write_text(ARM_XML)
CALL = dict(position=REACHABLE, tol=0.05, max_steps=400)

def png_to_arr(r):
    import io
    from PIL import Image
    assert r.get("status") == "success", r
    b = next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))

def build(n, cam_pos, cam_tgt, fov):
    s = Simulation(tool_name="art", mesh=False)
    assert s.create_world(gravity=[0, 0, 0])["status"] == "success"
    for i in range(n):
        name = "arm" if i == 0 else f"arm{i+1}"
        assert s.add_robot(name, urdf_path=str(xml), position=[0.8 * i, 0.0, 0.0])["status"] == "success"
    assert s.add_camera(name="look", position=cam_pos, target=cam_tgt, fov=fov)["status"] == "success"
    return s

def shot(s):
    return png_to_arr(s.render(camera_name="look", width=760, height=680))

def text(r):
    return " ".join(c["text"] for c in r.get("content", []) if "text" in c)

facts = {"tree": str(pathlib.Path(strands_robots.__file__).parents[1])}

# --- sole robot: omitted name resolves it ---
s1 = build(1, [0.36, -0.38, 0.34], [0.08, 0.05, 0.22], 30)
sole_home = shot(s1)
r_omit = s1.move_to(**CALL)
sole_after = shot(s1)
facts["sole_omitted"] = {"status": r_omit["status"], "text": text(r_omit)}
s1.cleanup(policy_stop_timeout=2.0)

# --- sole robot: naming it explicitly (fresh identical world) ---
s2 = build(1, [0.36, -0.38, 0.34], [0.08, 0.05, 0.22], 30)
r_named = s2.move_to(robot_name="arm", **CALL)
named_after = shot(s2)
facts["sole_named"] = {"status": r_named["status"], "text": text(r_named)}
facts["envelopes_identical"] = bool(r_omit == r_named)
s2.cleanup(policy_stop_timeout=2.0)

# --- ambiguous scene: omitted name is refused, nothing moves ---
s3 = build(2, [0.40, -1.15, 0.62], [0.40, 0.00, 0.20], 45)
amb_home = shot(s3)
r_amb = s3.move_to(**CALL)
amb_after = shot(s3)
facts["ambiguous_omitted"] = {"status": r_amb["status"], "text": text(r_amb)}
r_named2 = s3.move_to(robot_name="arm", **CALL)
facts["ambiguous_named"] = {"status": r_named2["status"], "text": text(r_named2)}
s3.cleanup(policy_stop_timeout=2.0)

def dmax(a, b):
    return int(np.abs(a.astype(int) - b.astype(int)).max())
def dfrac(a, b):
    return float((np.abs(a.astype(int) - b.astype(int)).sum(2) > 12).mean())

facts["metrics"] = {
    "omitted_vs_named_max_delta": dmax(sole_after, named_after),
    "sole_moved_frac": dfrac(sole_home, sole_after),
    "ambiguous_unmoved_max_delta": dmax(amb_home, amb_after),
}
for n, a in (("sole_home", sole_home), ("sole_after", sole_after),
             ("named_after", named_after), ("amb_home", amb_home), ("amb_after", amb_after)):
    np.save(OUT / f"{n}.npy", a)
(OUT / "facts.json").write_text(json.dumps(facts, indent=2))
print(json.dumps(facts["metrics"], indent=2))
print("sole omitted :", facts["sole_omitted"]["text"][:110])
print("ambiguous    :", facts["ambiguous_omitted"]["text"][:110])
print("envelopes identical:", facts["envelopes_identical"])
