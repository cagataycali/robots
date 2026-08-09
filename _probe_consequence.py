"""Measure what the caller actually gets when the example runs as written."""
import pathlib
import strands_robots.simulation.mujoco.simulation as m
print("TREE:", pathlib.Path(m.__file__).parents[3])
from strands_robots import Robot

def build(fixed: bool):
    sim = Robot("so101")
    params = {"position": [0.5, 0.0, 0.5], "target": [0.0, 0.0, 0.1], "width": 640, "height": 480}
    params["name" if fixed else "camera_name"] = "front"
    add = sim._dispatch_action("add_camera", params)
    cams = sim._dispatch_action("list_cameras", {})
    obs = sim.get_observation(robot_name="so101")
    imgs = sorted(k for k, v in obs.items() if hasattr(v, "shape"))
    return sim, {
        "add_status": add.get("status"),
        "cameras": cams if isinstance(cams, list) else cams.get("content"),
        "image_keys_in_obs": imgs,
        "n_image_keys": len(imgs),
    }

for fixed in (False, True):
    sim, facts = build(fixed)
    print(f"\n=== {'FIXED (name=)' if fixed else 'AS WRITTEN (camera_name=)'} ===")
    for k, v in facts.items():
        print(f"  {k}: {v}")
    sim.cleanup()
