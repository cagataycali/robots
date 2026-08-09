"""Capture what the policy is fed, as-written vs fixed. One tree: the only
variable is which parameter name the example passes to add_camera."""
import json, pathlib, sys
import numpy as np
import strands_robots.simulation.mujoco.simulation as m
print("TREE:", pathlib.Path(m.__file__).parents[3])
from strands_robots import Robot

OUT = pathlib.Path("/tmp/art244")
facts = {}

def frame_of(sim, cam):
    r = sim._dispatch_action("render", {"camera_name": cam, "width": 640, "height": 480})
    if r.get("status") != "success":
        raise RuntimeError(f"render({cam}) failed: {r}")
    import io
    from PIL import Image
    b = next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)
    return np.array(Image.open(io.BytesIO(b)).convert("RGB"))

for tag, key in (("as_written", "camera_name"), ("fixed", "name")):
    sim = Robot("so101")
    params = {"position": [0.5, 0.0, 0.5], "target": [0.0, 0.0, 0.1], "width": 640, "height": 480}
    params[key] = "front"
    add = sim._dispatch_action("add_camera", params)
    obs = sim.get_observation(robot_name="so101")
    imgs = sorted(k for k, v in obs.items() if hasattr(v, "shape"))
    # The view the policy actually receives: 'front' when it exists, else the
    # only camera the world has.
    served = "front" if "front" in imgs else imgs[0]
    fr = frame_of(sim, served)
    np.save(OUT / f"{tag}.npy", fr)
    facts[tag] = {
        "add_camera_param": key,
        "add_status": add.get("status"),
        "add_text": " | ".join(c.get("text", "") for c in add.get("content", []) if "text" in c)[:120],
        "image_keys": imgs,
        "camera_served_to_policy": served,
        "front_camera_exists": "front" in imgs,
    }
    sim.cleanup()

a, b = np.load(OUT / "as_written.npy"), np.load(OUT / "fixed.npy")
facts["diff_fraction"] = float((np.abs(a.astype(int) - b.astype(int)).sum(2) > 12).mean())
for t in ("as_written", "fixed"):
    f = np.load(OUT / f"{t}.npy")
    facts[t]["saturation"] = float(((f.max(2) - f.min(2)) > 45).mean())

assert facts["as_written"]["add_status"] == "error"
assert facts["fixed"]["add_status"] == "success"
assert facts["as_written"]["front_camera_exists"] is False
assert facts["fixed"]["front_camera_exists"] is True
assert facts["diff_fraction"] > 0.10, f"panels too similar: {facts['diff_fraction']:.4f}"
for t in ("as_written", "fixed"):
    assert facts[t]["saturation"] > 0.03, f"{t} looks blank: {facts[t]['saturation']:.4f}"
(OUT / "facts.json").write_text(json.dumps(facts, indent=2))
print(json.dumps(facts, indent=2))
