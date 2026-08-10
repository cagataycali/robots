"""Measure the unknown-entity report in whichever tree this file sits in."""
import json, pathlib, sys
import strands_robots.simulation.base as sbase
TREE = pathlib.Path(sbase.__file__).parents[2]
print("TREE:", TREE)
from strands_robots import Simulation
from strands_robots.simulation.base import SimEngine

ARM = """<mujoco model="arm"><compiler angle="radian"/>
 <visual><headlight ambient="0.55 0.55 0.55" diffuse="0.65 0.65 0.65"/></visual>
 <worldbody><body name="base" pos="0 0 0.05"><geom type="box" size="0.05 0.05 0.05" rgba=".35 .38 .45 1"/>
  <body name="link" pos="0 0 0.06"><joint name="pan" type="hinge" axis="0 0 1" range="-2 2" damping="4"/>
   <geom type="capsule" fromto="0 0 0 0.18 0 0" size="0.022" rgba=".25 .55 .85 1"/>
   <site name="tcp" pos="0.18 0 0" size="0.006"/></body></body></worldbody>
 <actuator><position name="a_pan" joint="pan" kp="50" ctrlrange="-2 2"/></actuator>
 <sensor><framepos name="tcp_pos" objtype="site" objname="tcp"/></sensor></mujoco>"""

out = {"tree": str(TREE)}
arm = pathlib.Path("_art/arm.xml"); arm.write_text(ARM)

sim = Simulation(backend="mujoco", mesh=False)
assert sim.create_world(gravity=[0, 0, -9.81])["status"] == "success"
assert sim.add_robot(name="arm", urdf_path=str(arm))["status"] == "success"
assert sim.add_object(name="crate", shape="box", size=[0.08, 0.08, 0.08],
                     position=[0.32, 0.0, 0.04], color=[0.95, 0.55, 0.12, 1])["status"] == "success"
assert sim.add_camera(name="look", position=[0.60, -0.52, 0.36], target=[0.12, 0, 0.10], fov=42)["status"] == "success"
assert sim.save_state(name="cp1")["status"] == "success"

BAD = ["front"]                       # what move_to's first positional lands
HELPERS = [
    ("Robot (facade)",  lambda: SimEngine._unknown_robot_msg(sim, BAD)),
    ("Robot (MuJoCo)",  lambda: sim._unknown_robot_msg(BAD)),
    ("Object",          lambda: sim._unknown_object_msg(BAD)),
    ("Camera",          lambda: sim._unknown_camera_msg(BAD)),
    ("Body",            lambda: sim._unknown_mj_entity_msg("Body", BAD)),
    ("Joint",           lambda: sim._unknown_mj_entity_msg("Joint", BAD)),
    ("Sensor",          lambda: sim._unknown_mj_entity_msg("Sensor", BAD)),
]
out["messages"] = {}
for label, fn in HELPERS:
    try:
        out["messages"][label] = {"ok": True, "text": fn()}
    except BaseException as e:   # noqa: BLE001 - an escape is one of the answers
        out["messages"][label] = {"ok": False, "text": f"RAISED {type(e).__name__}: {e}"}

# The two lookups that sat in front of a report.
out["reached"] = {}
for label, fn in [("get_sensor_data", lambda: sim.get_sensor_data(BAD)),
                  ("load_state", lambda: sim.load_state(BAD))]:
    try:
        r = fn()
        t = next((c["text"] for c in r.get("content", []) if "text" in c), "")
        out["reached"][label] = {"ok": True, "status": r.get("status"), "text": t}
    except BaseException as e:   # noqa: BLE001
        out["reached"][label] = {"ok": False, "status": "RAISED", "text": f"{type(e).__name__}: {e}"}

# A str typo, which must be untouched.
out["str_typo"] = {label: fn.__self__ if False else None for label, fn in []}
out["str_typo"] = {
    "Robot (MuJoCo)": sim._unknown_robot_msg("arm0"),
    "Object": sim._unknown_object_msg("crat"),
    "Body": sim._unknown_mj_entity_msg("Body", "arm/bas"),
}

# The honored path still runs: settle, then one real headless render.
for _ in range(60):
    sim.send_action({"a_pan": 0.9}, robot_name="arm", n_substeps=10)
st = sim.get_body_state("arm/link")
out["joint"] = next(c["json"] for c in st["content"] if "json" in c)["position"]
r = sim.render(camera_name="look", width=560, height=460)
assert r["status"] == "success", r
png = next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)
tag = "branch" if "close_match_hint" in pathlib.Path(sbase.__file__).read_text() else "main"
pathlib.Path(f"/tmp/art_render_{tag}.png").write_bytes(png)
out["tag"] = tag
pathlib.Path(f"/tmp/art_{tag}.json").write_text(json.dumps(out, indent=1))
print("tag:", tag, "joint:", out["joint"])
sim.cleanup()
