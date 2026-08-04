import os, sys; os.environ["MUJOCO_GL"]="egl"
import mujoco as mj
from strands_robots.simulation import Simulation
def jj(r): return next((c["json"] for c in r.get("content",[]) if "json" in c), None)
def txt(r): return next((c["text"] for c in r.get("content",[]) if "text" in c), "")

CUBE = 0.030
PAD_BELOW_HAND = 0.098

def pad_sep(sim):
    m, d = sim._world._model, sim._world._data
    ys = [(float(d.geom_xpos[g][1]), float(m.geom_size[g][1])) for g in range(m.ngeom)
          if "finger" in (mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "")
          and mj.mjtGeom(int(m.geom_type[g])).name == "mjGEOM_BOX"
          and abs(float(m.geom_size[g][0]) - 0.0085) < 1e-6]
    if len(ys) != 2: return None
    (y1,t1),(y2,t2) = ys
    return abs(y1-y2)-t1-t2

for TOP, X, grip in [(0.12, 0.50, 0.52), (0.16, 0.48, 0.52), (0.12, 0.45, 0.52)]:
    CZ = TOP + CUBE/2
    HAND = CZ + PAD_BELOW_HAND
    sim = Simulation(backend="mujoco", tool_name="c3", mesh=False)
    try:
        sim.create_world(); sim.add_robot(name="arm", data_config="panda")
        sim.add_object(name="table", shape="box", size=[0.20,0.26,TOP], position=[X,0.0,TOP/2],
                       is_static=True, color=[0.58,0.60,0.64,1])
        sim.add_object(name="cube", shape="box", size=[CUBE]*3, position=[X,0.0,CZ],
                       mass=0.05, color=[0.95,0.38,0.08,1])
        sim.step(200)
        z0 = jj(sim.get_body_state(body_name="cube"))["position"]
        for _ in range(40): sim.send_action({"actuator8":1.0}, robot_name="arm", n_substeps=8)
        h  = sim.move_to(robot_name="arm", position=[X,0.0,HAND+0.13], tol=0.02, max_steps=500)
        dd = sim.move_to(robot_name="arm", position=[X,0.0,HAND], tol=0.015, max_steps=600)
        if dd["status"] != "success":
            print(f"TOP={TOP} X={X}: hover={h['status']} descend=ERROR: {txt(dd)[:150]}", flush=True); continue
        for _ in range(110): sim.send_action({"actuator8":grip}, robot_name="arm", n_substeps=8)
        sep = pad_sep(sim)
        L = sim.move_to(robot_name="arm", position=[X,0.0,HAND+0.20], tol=0.02, max_steps=700)
        zF = jj(sim.get_body_state(body_name="cube"))["position"]
        print(f"TOP={TOP} X={X} grip={grip}: {h['status']}/{dd['status']}/{L['status']} pad_sep={sep:.4f} "
              f"| cube z {z0[2]:.4f} -> {zF[2]:.4f} | LIFT={zF[2]-z0[2]:+.4f}", flush=True)
    finally:
        sim.cleanup()
