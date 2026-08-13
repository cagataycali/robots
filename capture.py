"""Capture: an honored primitive moves the arm; a resolution refusal costs nothing."""
import io, json, pathlib, sys
import numpy as np
import strands_robots

ROOT = pathlib.Path(strands_robots.__file__).parents[1]
print("TREE:", ROOT)
OUT = pathlib.Path(__file__).resolve().parent
RUN = sys.argv[1]
FACTS = pathlib.Path(f"/tmp/art-facts-{RUN}.json")
facts = {"tree": str(ROOT)}


def save():
    FACTS.write_text(json.dumps(facts, indent=2))


sys.path.insert(0, str(ROOT / "tests" / "simulation" / "mujoco"))
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

sys.path.insert(0, str(ROOT))
from tests.simulation.mujoco.test_motion_primitives import ARM_XML, REACHABLE  # noqa: E402
from tests.simulation.mujoco.test_primitive_resolution_refusals import (  # noqa: E402
    ACTUATORLESS_XML,
    GRIPPER_ONLY_XML,
)

tmp = pathlib.Path(f"/tmp/artscene-{RUN}")
tmp.mkdir(parents=True, exist_ok=True)
W, H = 620, 560


def render(sim, cam):
    r = sim.render(camera_name=cam, width=W, height=H)
    assert r.get("status") == "success", r
    png = next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)
    import imageio.v3 as iio

    return np.asarray(iio.imread(io.BytesIO(png)))[:, :, :3]


def metrics(a, b):
    d = np.abs(a.astype(int) - b.astype(int)).max(2)
    return float((d > 8).mean()), int(d.max())


def arm_frac(img):
    return float((img.max(2).astype(int) - img.min(2).astype(int) > 45).mean())


def text_of(res):
    return "\n".join(c["text"] for c in res.get("content", []) if "text" in c)


def scene(label, xml, cam_pos, cam_tgt, fov):
    p = tmp / f"{label}.xml"
    p.write_text(xml)
    s = Simulation(tool_name=f"art_{label}", mesh=False)
    assert s.create_world(gravity=[0, 0, 0])["status"] == "success"
    assert s.add_robot("arm", urdf_path=str(p))["status"] == "success"
    assert s.add_camera(name="look", position=cam_pos, target=cam_tgt, fov=fov)["status"] == "success"
    return s


# ---- 1. conventional arm: the honored move_to actually moves it -------------
sim = scene("conventional", ARM_XML, [0.36, -0.38, 0.34], [0.08, 0.05, 0.22], 30)
home = render(sim, "look")
res = sim.move_to(robot_name="arm", position=list(REACHABLE), tol=0.02, max_steps=400)
after = render(sim, "look")
moved, dmax = metrics(home, after)
facts["honored"] = {
    "status": res["status"],
    "text": text_of(res)[:200],
    "moved_frac": moved,
    "max_delta": dmax,
    "arm_frac": arm_frac(home),
}
print("honored:", facts["honored"])
assert res["status"] == "success", res
assert moved > 0.10, f"honored move_to only changed {moved:.2%} of pixels - reframe"
assert arm_frac(home) > 0.15, f"arm is only {arm_frac(home):.2%} of frame - reframe"
np.save(OUT / "p_home.npy", home)
np.save(OUT / "p_honored.npy", after)
sim.cleanup(policy_stop_timeout=2.0)
save()

# ---- 2. actuatorless arm: move_to refuses and the scene is untouched --------
sim = scene("actuatorless", ACTUATORLESS_XML, [0.30, -0.30, 0.26], [0.07, 0.0, 0.07], 34)
h2 = render(sim, "look")
r2 = sim.move_to(robot_name="arm", position=list(REACHABLE), tol=0.02, max_steps=400)
a2 = render(sim, "look")
ch2, dm2 = metrics(h2, a2)
facts["refused_move_to"] = {
    "status": r2["status"],
    "text": text_of(r2),
    "changed_frac": ch2,
    "max_delta": dm2,
    "changed_px": int((np.abs(h2.astype(int) - a2.astype(int)).max(2) > 8).sum()),
    "arm_frac": arm_frac(h2),
}
print("refused move_to:", {k: v for k, v in facts["refused_move_to"].items() if k != "text"})
assert r2["status"] == "error"
assert facts["refused_move_to"]["changed_px"] == 0, "a refusal must cost no pixels"
np.save(OUT / "p_refused_move.npy", a2)
sim.cleanup(policy_stop_timeout=2.0)
save()

# ---- 3. gripper-only arm: rotate_wrist refuses, scene untouched -------------
sim = scene("gripper_only", GRIPPER_ONLY_XML, [0.14, -0.14, 0.13], [0.0, 0.0, 0.06], 40)
h3 = render(sim, "look")
r3 = sim.rotate_wrist(robot_name="arm", target_yaw=0.3)
a3 = render(sim, "look")
ch3, dm3 = metrics(h3, a3)
facts["refused_rotate_wrist"] = {
    "status": r3["status"],
    "text": text_of(r3),
    "changed_px": int((np.abs(h3.astype(int) - a3.astype(int)).max(2) > 8).sum()),
    "max_delta": dm3,
    "arm_frac": arm_frac(h3),
}
print("refused rotate_wrist:", {k: v for k, v in facts["refused_rotate_wrist"].items() if k != "text"})
assert r3["status"] == "error"
assert facts["refused_rotate_wrist"]["changed_px"] == 0
np.save(OUT / "p_refused_wrist.npy", a3)
sim.cleanup(policy_stop_timeout=2.0)
save()
print("OK ->", FACTS)
