"""Probe: which of the example's two dispatch calls the router accepts."""
import pathlib
import strands_robots.simulation.mujoco.simulation as m
print("TREE:", pathlib.Path(m.__file__).parents[3])

from strands_robots import Robot

sim = Robot("so101")
rows = []

for action, params, label in [
    ("add_camera", {"camera_name": "front", "position": [0.5, 0.0, 0.5],
                    "target": [0.0, 0.0, 0.1], "width": 640, "height": 480}, "line 52 as written"),
    ("add_camera", {"name": "front", "position": [0.5, 0.0, 0.5],
                    "target": [0.0, 0.0, 0.1], "width": 640, "height": 480}, "line 52 fixed"),
]:
    try:
        r = sim._dispatch_action(action, params)
        st = r.get("status")
        txt = " | ".join(c.get("text", "") for c in r.get("content", []) if "text" in c)
    except BaseException as e:
        st, txt = f"RAISED {type(e).__name__}", str(e)
    rows.append((label, action, st, txt[:150]))

for params, label in [
    ({"camera_name": "front"}, "line 77 as written"),
    ({}, "line 77 without the param"),
]:
    try:
        r = sim._dispatch_action("get_observation", params)
        st = r.get("status")
        txt = " | ".join(c.get("text", "") for c in r.get("content", []) if "text" in c)
    except BaseException as e:
        st, txt = f"RAISED {type(e).__name__}", str(e)
    rows.append((label, "get_observation", st, txt[:150]))

for label, action, st, txt in rows:
    print(f"\n[{label}]  {action}\n  status={st}\n  {txt}")
sim.cleanup()
