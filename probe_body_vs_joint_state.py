"""Do the BODIES move while joint_q reports nothing?"""
import numpy as np, pathlib, json
import strands_robots
print("TREE:", pathlib.Path(strands_robots.__file__).parents[1], flush=True)
from strands_robots.simulation import create_simulation
from strands_robots.simulation.newton import simulation as nsim
nsim.articulated_solver_error = lambda _s: None

TARGET = {"j1": 0.9, "j2": -0.7}
facts = {}
for solver in ("featherstone", "xpbd", "semi_implicit"):
    sim = create_simulation("newton", solver=solver, mesh=False)
    row = {}
    facts[solver] = row
    try:
        sim.create_world()
        sim.add_robot(name="arm", urdf_path="_probe/probe_arm.xml")
        m = sim._model
        bq0 = np.array(m.body_q.numpy(), copy=True)
        jq0 = np.array(sim._state_0.joint_q.numpy(), copy=True)
        sim.send_action(TARGET, robot_name="arm", n_substeps=1)
        sim.step(120)
        bq1 = np.array(m.body_q.numpy(), copy=True)
        jq1 = np.array(sim._state_0.joint_q.numpy(), copy=True)
        row["body_pos_before"] = [[round(float(v), 4) for v in b[:3]] for b in bq0]
        row["body_pos_after"] = [[round(float(v), 4) for v in b[:3]] for b in bq1]
        row["max_body_shift_m"] = round(float(np.abs(bq1[:, :3] - bq0[:, :3]).max()), 6)
        row["joint_q_before"] = [round(float(v), 5) for v in jq0]
        row["joint_q_after"] = [round(float(v), 5) for v in jq1]
        row["max_joint_shift"] = round(float(np.abs(jq1 - jq0).max()), 6)
    except BaseException as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"[:200]
    finally:
        sim.cleanup()
    print(f"{solver:14s} max_body_shift={row.get('max_body_shift_m')}m "
          f"max_joint_shift={row.get('max_joint_shift')} "
          f"bodies_after={row.get('body_pos_after')}", flush=True)
pathlib.Path("_probe/facts_bodies.json").write_text(json.dumps(facts, indent=2))
print("DONE", flush=True)
