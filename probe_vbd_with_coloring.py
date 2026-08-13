"""Does the issue's prescribed remedy (builder.color() before finalize) make VBD simulate?

Models the fix faithfully by wrapping ModelBuilder.finalize so color() runs first,
then measures the same physical outcome the solver matrix used: does a commanded
joint actually move?
"""
import json, pathlib, time, traceback

import strands_robots
TREE = pathlib.Path(strands_robots.__file__).parents[1]
print("TREE:", TREE, flush=True)
import newton
from strands_robots.simulation import create_simulation

orig_finalize = newton.ModelBuilder.finalize
colored = []


def patched_finalize(self, *args, **kwargs):
    self.color()  # the issue's prescribed minimal fix
    colored.append({"bodies": int(self.body_count),
                    "body_color_groups": [len(g) for g in self.body_color_groups],
                    "particle_color_groups": len(self.particle_color_groups)})
    return orig_finalize(self, *args, **kwargs)


newton.ModelBuilder.finalize = patched_finalize

TARGET = {"j1": 0.9, "j2": -0.7}
facts = {"tree": str(TREE), "rows": {}}
out = pathlib.Path("_probe/facts_vbd_fix.json")

for name in ("vbd", "style3d"):
    row = {"solver": name}
    facts["rows"][name] = row
    sim = None
    t0 = time.time()
    try:
        sim = create_simulation("newton", solver=name, mesh=False)
        sim.create_world()
        r = sim.add_robot(name="arm", urdf_path="_probe/probe_arm.xml")
        row["add_robot"] = r.get("status")
        row["add_robot_text"] = str(r)[:250]
        if r.get("status") == "success":
            before = {j: float(sim.get_observation(robot_name="arm")[j]) for j in TARGET}
            sim.send_action(TARGET, robot_name="arm", n_substeps=1)
            st = sim.step(120)
            row["step"] = st.get("status")
            after = {j: float(sim.get_observation(robot_name="arm")[j]) for j in TARGET}
            row["max_travel"] = round(max(abs(after[j] - before[j]) for j in TARGET), 6)
            row["moved"] = row["max_travel"] > 0.05
            row["verdict"] = "moves" if row["moved"] else "builds but does not move"
        else:
            row["verdict"] = "add_robot refused"
    except BaseException as exc:  # noqa: BLE001 - classify every failure mode
        row["verdict"] = type(exc).__name__
        row["error"] = str(exc)[:400]
        row["tb_tail"] = traceback.format_exc().strip().splitlines()[-1][:200]
    finally:
        row["elapsed_s"] = round(time.time() - t0, 1)
        if sim is not None:
            try:
                sim.cleanup()
            except BaseException:  # noqa: BLE001 - best-effort teardown
                pass
        facts["coloring_calls"] = colored
        out.write_text(json.dumps(facts, indent=2))
        print(f"{name:10s} {row.get('verdict'):36s} travel={row.get('max_travel')} "
              f"{row.get('elapsed_s')}s", flush=True)

print("coloring:", colored, flush=True)
print("DONE", flush=True)
