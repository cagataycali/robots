import os, sys; os.environ["MUJOCO_GL"]="egl"
import mujoco as mj
from strands_robots.simulation import Simulation
sim = Simulation(backend="mujoco", tool_name="opt", mesh=False)
try:
    sim.create_world(); sim.add_robot(name="arm", data_config="panda")
    m = sim._world._model
    print(f"{sys.argv[1]}: integrator={mj.mjtIntegrator(int(m.opt.integrator)).name} "
          f"cone={mj.mjtCone(int(m.opt.cone)).name} impratio={m.opt.impratio:.2f} timestep={m.opt.timestep}")
finally:
    sim.cleanup()
