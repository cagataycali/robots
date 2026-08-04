import os
os.environ["MUJOCO_GL"] = "egl"
from strands_robots.simulation import Simulation

def j(res):
    return next((c["json"] for c in res.get("content", []) if "json" in c), None)

sim = Simulation(backend="mujoco", tool_name="pick", mesh=False)
try:
    print("world:", sim.create_world()["status"])
    print("robot:", sim.add_robot(name="arm", data_config="panda")["status"])
    print("cube :", sim.add_object(name="cube", shape="box", size=[0.04,0.04,0.04],
                                   position=[0.5,0.0,0.02], mass=0.05,
                                   color=[0.95,0.35,0.08,1])["status"])
    sim.step(150)
    print("cube z0:", j(sim.get_body_state(body_name="cube"))["position"])
    r = sim.set_gripper(robot_name="arm", state="open")
    print("open:", r["status"], str(r)[:120])
    r = sim.move_to(robot_name="arm", position=[0.5, 0.0, 0.22], tol=0.02, max_steps=400)
    print("hover:", r["status"], str(j(r))[:220])
    r = sim.move_to(robot_name="arm", position=[0.5, 0.0, 0.128], tol=0.015, max_steps=400)
    print("descend:", r["status"], str(j(r))[:220])
    r = sim.set_gripper(robot_name="arm", state="close", steps=60)
    print("close:", r["status"], str(j(r))[:180])
    r = sim.move_to(robot_name="arm", position=[0.5, 0.0, 0.30], tol=0.02, max_steps=500)
    print("lift:", r["status"], str(j(r))[:220])
    print("cube zf:", j(sim.get_body_state(body_name="cube"))["position"])
finally:
    sim.cleanup()
