import os, time
os.environ["MUJOCO_GL"] = "egl"
from strands_robots.simulation import Simulation

t0 = time.time()
sim = Simulation(backend="mujoco", tool_name="smoke", mesh=False)
print("create_world(terrain='rough'):", sim.create_world(terrain="rough", difficulty=1.0)["status"])
r = sim.add_robot(name="go2", data_config="go2")
print("add_robot go2:", r["status"], str(r)[:180])
print("ground_height(0,0):", sim.get_ground_height(0.0, 0.0))
print("ground_height(1.2,0.4):", sim.get_ground_height(1.2, 0.4))
print("cams:", sim.list_cameras())
print(f"elapsed {time.time()-t0:.1f}s")
sim.cleanup()
