import os; os.environ["MUJOCO_GL"]="egl"
import mujoco as mj
from strands_robots.simulation import Simulation
def jj(r): return next((c["json"] for c in r.get("content",[]) if "json" in c), None)

CUBE, TOP, X = 0.030, 0.12, 0.50
CZ = TOP + CUBE/2
HAND = CZ + 0.098

def pad_sep(sim):
    m,d = sim._world._model, sim._world._data
    ys=[(float(d.geom_xpos[g][1]), float(m.geom_size[g][1])) for g in range(m.ngeom)
        if "finger" in (mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,m.geom_bodyid[g]) or "")
        and mj.mjtGeom(int(m.geom_type[g])).name=="mjGEOM_BOX"
        and abs(float(m.geom_size[g][0])-0.0085)<1e-6]
    (y1,t1),(y2,t2)=ys; return abs(y1-y2)-t1-t2

def cube_contacts(sim):
    r = sim.get_contacts()
    js = jj(r) or {}
    cs = js.get("contacts", [])
    out = []
    for c in cs:
        names = f"{c.get('geom1','')}|{c.get('geom2','')}"
        if "cube" in names:
            out.append((names, round(float(c.get("normal_force", 0.0)), 4), c.get("active")))
    return out

for grip in (0.44, 0.36, 0.28):
    sim = Simulation(backend="mujoco", tool_name="c4", mesh=False)
    try:
        sim.create_world(); sim.add_robot(name="arm", data_config="panda")
        sim.add_object(name="table", shape="box", size=[0.20,0.26,TOP], position=[X,0.0,TOP/2],
                       is_static=True, color=[0.58,0.60,0.64,1])
        sim.add_object(name="cube", shape="box", size=[CUBE]*3, position=[X,0.0,CZ],
                       mass=0.05, color=[0.95,0.38,0.08,1])
        sim.step(200)
        z0 = jj(sim.get_body_state(body_name="cube"))["position"]
        for _ in range(40): sim.send_action({"actuator8":1.0}, robot_name="arm", n_substeps=8)
        sim.move_to(robot_name="arm", position=[X,0.0,HAND+0.13], tol=0.02, max_steps=500)
        dd = sim.move_to(robot_name="arm", position=[X,0.0,HAND], tol=0.015, max_steps=600)
        for _ in range(140): sim.send_action({"actuator8":grip}, robot_name="arm", n_substeps=8)
        sep = pad_sep(sim); con = cube_contacts(sim)
        zG = jj(sim.get_body_state(body_name="cube"))["position"]
        L = sim.move_to(robot_name="arm", position=[X,0.0,HAND+0.18], tol=0.025, max_steps=900)
        # keep holding while the arm settles at the top
        for _ in range(120): sim.send_action({"actuator8":grip}, robot_name="arm", n_substeps=8)
        zF = jj(sim.get_body_state(body_name="cube"))["position"]
        print(f"grip={grip} sep={sep:.4f} squeeze={CUBE-sep:.4f} lift={L['status']} "
              f"| z {z0[2]:.4f}->{zG[2]:.4f}->{zF[2]:.4f} LIFT={zF[2]-z0[2]:+.4f}")
        print(f"   cube contacts at grasp: {con}", flush=True)
    finally:
        sim.cleanup()
