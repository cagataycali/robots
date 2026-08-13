"""Gravity-only: does the solver integrate rigid bodies at all (no target commanded)?"""
import json, pathlib, time, traceback
import strands_robots
print("TREE:", pathlib.Path(strands_robots.__file__).parents[1], flush=True)
from strands_robots.simulation import create_simulation

JOINTS = ("j1", "j2")
facts = {"rows": {}}
out = pathlib.Path("_probe/facts_gravity.json")
for name in ("mujoco", "xpbd", "semi_implicit"):
    row = {}
    facts["rows"][name] = row
    sim = None
    try:
        sim = create_simulation("newton", solver=name, mesh=False)
        sim.create_world()
        sim.add_robot(name="arm", urdf_path="_probe/probe_arm.xml")
        before = {j: float(sim.get_observation(robot_name="arm")[j]) for j in JOINTS}
        sim.step(200)  # gravity only, nothing commanded
        after = {j: float(sim.get_observation(robot_name="arm")[j]) for j in JOINTS}
        row["before"], row["after"] = before, after
        row["gravity_travel"] = round(max(abs(after[j] - before[j]) for j in JOINTS), 6)
        row["integrates"] = row["gravity_travel"] > 1e-4
        st = sim.get_state()
        row["state_text"] = str(st)[:150]
    except BaseException as exc:  # noqa: BLE001 - classify every failure mode
        row["error"] = f"{type(exc).__name__}: {exc}"[:300]
    finally:
        if sim is not None:
            try:
                sim.cleanup()
            except BaseException:  # noqa: BLE001 - best-effort teardown
                pass
        out.write_text(json.dumps(facts, indent=2))
        print(f"{name:14s} gravity_travel={row.get('gravity_travel')} integrates={row.get('integrates')} {row.get('error','')}", flush=True)
print("DONE", flush=True)
