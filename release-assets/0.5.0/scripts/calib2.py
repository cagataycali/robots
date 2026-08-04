import os; os.environ["MUJOCO_GL"]="egl"
import mujoco as mj
from strands_robots.simulation import Simulation
def jj(r): return next((c["json"] for c in r.get("content",[]) if "json" in c), None)

CUBE, TOP = 0.030, 0.30
CZ   = TOP + CUBE/2          # 0.315
PAD_BELOW_HAND = 0.098
HAND = CZ + PAD_BELOW_HAND   # 0.413

def pad_sep(sim):
    m, d = sim._world._model, sim._world._data
    ys = []
    for g in range(m.ngeom):
        bn = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
        if "finger" in bn and mj.mjtGeom(int(m.geom_type[g])).name == "mjGEOM_BOX" \
           and abs(float(m.geom_size[g][0]) - 0.0085) < 1e-6:
            ys.append((float(d.geom_xpos[g][1]), float(m.geom_size[g][1])))
    if len(ys) != 2: return None
    (y1, t1), (y2, t2) = ys
    return abs(y1 - y2) - t1 - t2   # inner-face separation

for grip in (0.70, 0.64, 0.58):
    sim = Simulation(backend="mujoco", tool_name="c2", mesh=False)
    try:
        sim.create_world(); sim.add_robot(name="arm", data_config="panda")
        sim.add_object(name="table", shape="box", size=[0.22,0.30,TOP], position=[0.52,0.0,TOP/2],
                       is_static=True, color=[0.58,0.60,0.64,1])
        sim.add_object(name="cube", shape="box", size=[CUBE]*3, position=[0.52,0.0,CZ],
                       mass=0.05, color=[0.95,0.38,0.08,1])
        sim.step(200)
        z0 = jj(sim.get_body_state(body_name="cube"))["position"]
        for _ in range(40): sim.send_action({"actuator8":1.0}, robot_name="arm", n_substeps=8)
        print(f"  open pad inner-face sep = {pad_sep(sim):.4f} m (cube is {CUBE} m)")
        h = sim.move_to(robot_name="arm", position=[0.52,0.0,HAND+0.14], tol=0.02, max_steps=500)
        d_ = sim.move_to(robot_name="arm", position=[0.52,0.0,HAND], tol=0.012, max_steps=500)
        zA = jj(sim.get_body_state(body_name="cube"))["position"]
        for _ in range(100): sim.send_action({"actuator8":grip}, robot_name="arm", n_substeps=8)
        sep = pad_sep(sim)
        L = sim.move_to(robot_name="arm", position=[0.52,0.0,HAND+0.20], tol=0.02, max_steps=700)
        zF = jj(sim.get_body_state(body_name="cube"))["position"]
        print(f"grip={grip}: {h['status']}/{d_['status']}/{L['status']} | pad_sep={sep:.4f} "
              f"| cube z {z0[2]:.4f} -> {zA[2]:.4f} -> {zF[2]:.4f} | LIFT={zF[2]-z0[2]:+.4f} "
              f"| drift={((zF[0]-z0[0])**2+(zF[1]-z0[1])**2)**0.5:.4f}", flush=True)
    finally:
        sim.cleanup()
