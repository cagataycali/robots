import os; os.environ["MUJOCO_GL"]="egl"
from strands_robots.simulation import Simulation
def jj(r): return next((c["json"] for c in r.get("content",[]) if "json" in c), None)

CUBE = 0.030
TOP  = 0.30
CZ   = TOP + CUBE/2
HAND = CZ + 0.058

for grip in (0.34, 0.28, 0.22):
    sim = Simulation(backend="mujoco", tool_name="cal", mesh=False)
    try:
        sim.create_world(); sim.add_robot(name="arm", data_config="panda")
        sim.add_object(name="table", shape="box", size=[0.22,0.30,TOP], position=[0.52,0.0,TOP/2],
                       is_static=True, color=[0.58,0.60,0.64,1])
        sim.add_object(name="cube", shape="box", size=[CUBE]*3, position=[0.52,0.0,CZ],
                       mass=0.05, color=[0.95,0.38,0.08,1])
        sim.step(200)
        z0 = jj(sim.get_body_state(body_name="cube"))["position"]
        for _ in range(40): sim.send_action({"actuator8":1.0}, robot_name="arm", n_substeps=8)
        h = sim.move_to(robot_name="arm", position=[0.52,0.0,HAND+0.12], tol=0.02, max_steps=500)
        d = sim.move_to(robot_name="arm", position=[0.52,0.0,HAND], tol=0.012, max_steps=500)
        zA = jj(sim.get_body_state(body_name="cube"))["position"]
        for _ in range(90): sim.send_action({"actuator8":grip}, robot_name="arm", n_substeps=8)
        obs = sim.get_observation(robot_name="arm")
        gap = sum(float(v) for k,v in obs.items() if "finger_joint" in k and not k.endswith(".vel"))
        L = sim.move_to(robot_name="arm", position=[0.52,0.0,HAND+0.18], tol=0.02, max_steps=600)
        zF = jj(sim.get_body_state(body_name="cube"))["position"]
        print(f"grip={grip}: hover={h['status']} descend={d['status']} lift={L['status']} "
              f"| gap={gap:.4f} | cube z {z0[2]:.4f} -> approach {zA[2]:.4f} -> final {zF[2]:.4f} "
              f"| lifted={zF[2]-z0[2]:+.4f} | xy drift={((zF[0]-z0[0])**2+(zF[1]-z0[1])**2)**0.5:.4f}", flush=True)
    finally:
        sim.cleanup()
