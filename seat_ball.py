import os, pathlib, math
os.environ.setdefault("MUJOCO_GL","egl")
import numpy as np, mujoco
from PIL import Image
from strands_robots.simulation import create_simulation
from strands_robots.assets import get_search_paths
scene = next(pathlib.Path(p)/"microduck"/"scene_ball.xml" for p in get_search_paths() if (pathlib.Path(p)/"microduck"/"scene_ball.xml").exists())
sim = create_simulation("mujoco"); sim.create_world()
print(sim.add_robot("microduck", urdf_path=str(scene))["status"])
m, d = sim._world._model, sim._world._data
def ball():
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "microduck/ball_free"); a=int(m.jnt_qposadr[j]); return [round(float(x),3) for x in d.qpos[a:a+7]]
def shot(name):
    r = mujoco.Renderer(m, 360, 480); cam = mujoco.MjvCamera(); cam.lookat[:] = [0.15, 0.0, 0.05]; cam.distance=0.7; cam.azimuth=150; cam.elevation=-25
    r.update_scene(d, cam); img = r.render(); Image.fromarray(img).save(name); r.close(); return img
print("before", ball()); a = shot("/tmp/before.png")
# trunk yaw frame offset (0.09 ahead, 0.042 side): trunk at reset is yaw 0
r = sim.set_joint_positions({"ball_free": [0.09, 0.042, 0.035, 1, 0, 0, 0]}, robot_name="microduck"); print(r["status"], r["content"][0]["text"])
r = sim.set_joint_velocities({"ball_free": [0.0]*6}, robot_name="microduck"); print(r["status"], r["content"][0]["text"])
print("after", ball()); b = shot("/tmp/after.png")
r = sim.set_joint_positions({"ball_free": 0.09}, robot_name="microduck"); print(r["status"], r["content"][0]["text"])
Image.fromarray(np.concatenate([a, b], axis=1)).save("/tmp/seat_ball_before_after.png")
