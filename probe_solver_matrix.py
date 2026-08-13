"""Which registered Newton solvers can build AND step an articulated robot?

Run once per tree. Records, per solver: construction, finalize, solver object,
step, and whether a commanded joint actually MOVES (the physical outcome).
"""
import json, os, pathlib, sys, time, traceback

import strands_robots
TREE = pathlib.Path(strands_robots.__file__).parents[1]
print("TREE:", TREE, flush=True)
from strands_robots.simulation import create_simulation
from strands_robots.simulation.newton.backend import solver_registry

ARM = """<mujoco model="probe_arm">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <body name="base" pos="0 0 0.05">
      <geom type="box" size="0.04 0.04 0.05"/>
      <body name="link1" pos="0 0 0.06">
        <joint name="j1" type="hinge" axis="0 0 1" range="-2 2" limited="true" damping="0.4"/>
        <geom type="capsule" fromto="0 0 0 0.14 0 0" size="0.02"/>
        <body name="link2" pos="0.15 0 0">
          <joint name="j2" type="hinge" axis="0 1 0" range="-2 2" limited="true" damping="0.4"/>
          <geom type="capsule" fromto="0 0 0 0.12 0 0" size="0.018"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a1" joint="j1" kp="30" ctrlrange="-2 2"/>
    <position name="a2" joint="j2" kp="30" ctrlrange="-2 2"/>
  </actuator>
</mujoco>
"""
xml_path = pathlib.Path("_probe/probe_arm.xml")
xml_path.write_text(ARM)

TARGET = {"j1": 0.9, "j2": -0.7}
rows = {}
out = pathlib.Path("_probe/facts_matrix.json")


def save():
    out.write_text(json.dumps({"tree": str(TREE), "rows": rows}, indent=2))


for name in sorted(solver_registry()):
    row = {"solver": name}
    rows[name] = row
    sim = None
    t0 = time.time()
    try:
        sim = create_simulation("newton", solver=name, mesh=False)
        row["construct"] = "ok"
        r = sim.create_world()
        row["create_world"] = r.get("status")
        r = sim.add_robot(name="arm", urdf_path=str(xml_path))
        row["add_robot"] = r.get("status")
        row["add_robot_text"] = str(r)[:220]
        if row["add_robot"] != "success":
            row["verdict"] = "add_robot refused"
            save(); continue
        before = {j: float(sim.get_observation(robot_name="arm")[j]) for j in TARGET}
        row["joints_before"] = before
        sr = sim.send_action(TARGET, robot_name="arm", n_substeps=1)
        row["send_action"] = sr.get("status")
        st = sim.step(120)
        row["step"] = st.get("status")
        after = {j: float(sim.get_observation(robot_name="arm")[j]) for j in TARGET}
        row["joints_after"] = after
        row["travel"] = {j: round(abs(after[j] - before[j]), 6) for j in TARGET}
        row["max_travel"] = round(max(row["travel"].values()), 6)
        row["moved"] = row["max_travel"] > 0.05
        row["verdict"] = "moves" if row["moved"] else "builds but does not move"
    except BaseException as exc:  # noqa: BLE001 - classify every failure mode
        row["verdict"] = f"{type(exc).__name__}"
        row["error"] = str(exc)[:400]
        row["traceback_tail"] = traceback.format_exc().strip().splitlines()[-1][:250]
    finally:
        row["elapsed_s"] = round(time.time() - t0, 1)
        if sim is not None:
            try:
                sim.cleanup()
            except BaseException:  # noqa: BLE001 - teardown is best effort here
                pass
        save()
        print(f"{name:14s} {row.get('verdict'):40s} {row.get('elapsed_s')}s "
              f"travel={row.get('max_travel')}", flush=True)

print("DONE", flush=True)
