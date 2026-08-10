"""Capture the observable state of a mounted camera across a remove_object call."""
import json, pathlib, sys
import numpy as np
import strands_robots
from strands_robots.simulation import Simulation

TREE = str(pathlib.Path(strands_robots.__file__).parents[1])
print("TREE:", TREE)
tag = sys.argv[1]
out = pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
W, H = 460, 380

def png(sim, cam):
    r = sim.render(camera_name=cam, width=W, height=H)
    if r.get("status") != "success":
        return None, r["content"][0]["text"]
    import io, imageio.v3 as iio
    b = next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)
    return np.asarray(iio.imread(io.BytesIO(b))[:, :, :3]), None

def verdicts(sim):
    v = {}
    for cam in ("watch", "plate_cam", "fixed"):
        listed = cam in sim.list_cameras()
        try:
            sim.get_camera_params(camera_name=cam); resolvable, msg = True, ""
        except KeyError as e:
            resolvable, msg = False, str(e).strip('"')
        v[cam] = {"listed": listed, "resolvable": resolvable, "message": msg}
    return v

sim = Simulation(backend="mujoco", tool_name="art", mesh=False)
sim.create_world(gravity=[0, 0, -9.81])
sim.add_object("crate", shape="box", size=[0.16, 0.16, 0.16], position=[0.28, 0.0, 0.08], mass=0.5,
               color=[0.95, 0.55, 0.12, 1.0])
sim.add_object("plate", shape="box", size=[0.30, 0.30, 0.02], position=[-0.26, 0.0, 0.01], is_static=True,
               color=[0.25, 0.55, 0.85, 1.0])
sim.add_camera(name="watch", position=[0.22, -0.26, 0.16], target=[0.0, 0.0, -0.02], parent_body="crate")
sim.add_camera(name="plate_cam", position=[0.0, -0.34, 0.26], target=[0.0, 0.0, -0.04], parent_body="plate")
sim.add_camera(name="fixed", position=[0.9, -0.9, 0.7], target=[0.0, 0.0, 0.1])
sim.step(60)

facts = {"tree": TREE, "tag": tag}
img, err = png(sim, "watch")
assert img is not None, err
np.save(out / f"watch_before_{tag}.npy", img)
facts["before"] = {"cameras": sim.list_cameras(), "ncam": int(sim._world._model.ncam), "verdicts": verdicts(sim)}

res = sim.remove_object("crate")
facts["remove_object"] = {"status": res["status"], "text": res["content"][0]["text"]}
facts["after"] = {"cameras": sim.list_cameras(), "ncam": int(sim._world._model.ncam), "verdicts": verdicts(sim)}
_, facts["after"]["watch_render_error"] = png(sim, "watch")
img2, err2 = png(sim, "plate_cam")
assert img2 is not None, err2
np.save(out / f"plate_after_{tag}.npy", img2)
(out / f"facts_{tag}.json").write_text(json.dumps(facts, indent=2))
print(json.dumps(facts, indent=2))
sim.cleanup()
