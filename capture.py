"""Measure what a caller gets for an unresolvable / mistyped policy_provider.

Run in a worktree at upstream/main and in the branch; each run records its own
tree so compose.py can prove the two halves came from different code.
"""
import json, os, pathlib, sys, time
import numpy as np
import strands_robots
TREE = str(pathlib.Path(strands_robots.__file__).parents[1])
print("TREE:", TREE)

OUT = pathlib.Path(sys.argv[1])
facts = {"tree": TREE, "rows": [], "honored": {}}
def save():
    OUT.write_text(json.dumps(facts, indent=1))

from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

SURFACES = [("run_policy", {"duration": 0.05}), ("eval_policy", {"n_episodes": 1, "max_steps": 2}),
            ("start_policy", {"duration": 0.3})]
VALUES = [("molmoact2", "unresolvable name"), ("MolmoAct2", "unresolvable name"),
          (None, "wrong type"), (3, "wrong type"),
          # Control: the trust-remote-code gate is a separate concern this change
          # deliberately leaves raising, so this row must look the same on both trees.
          ("hf.co/allenai/x", "trust gate (out of scope)")]

sim = MuJoCoSimEngine(tool_name="art", mesh=False)
sim.create_world()
sim.add_robot(name="so100", data_config="so100")
sim.add_camera(name="look", position=[0.62, -0.52, 0.42], target=[0.0, 0.0, 0.16], fov=42)

for val, klass in VALUES:
    for action, extra in SURFACES:
        row = {"action": action, "value": repr(val), "class": klass}
        try:
            r = getattr(sim, action)(robot_name="so100", policy_provider=val, **extra)
            row["status"] = r.get("status")
            row["text"] = " ".join(c["text"] for c in r.get("content", []) if "text" in c)[:200]
            row["raised"] = None
        except BaseException as e:  # noqa: BLE001 - the escape IS the finding
            row["status"] = None
            row["text"] = f"{type(e).__name__}: {e}"[:200]
            row["raised"] = type(e).__name__
        if action == "start_policy":
            time.sleep(0.45)
            lr = sim.list_policies_running()
            row["running_after"] = " ".join(c["text"] for c in lr.get("content", []) if "text" in c)[:120]
            try:
                sim.stop_policy(robot_name="so100")
            except BaseException:  # noqa: BLE001
                pass
            time.sleep(0.1)
        facts["rows"].append(row)
        print(f"  {action:13s} {row['value']:18s} status={row['status']!r:9s} raised={row['raised']}")
        save()

# The honored path must be untouched: a real mock rollout, rendered.
def frame():
    r = sim.render(camera_name="look", width=760, height=680)
    assert r.get("status") == "success", r
    png = next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)
    import imageio.v3 as iio
    return np.asarray(iio.imread(png))

before = frame()
res = sim.run_policy(robot_name="so100", policy_provider="mock", duration=1.6, control_frequency=50.0)
after = frame()
obs = sim.get_observation(robot_name="so100")
joints = {k: round(float(v), 6) for k, v in sorted(obs.items()) if not hasattr(v, "shape")}
facts["honored"] = {
    "status": res.get("status"),
    "text": " ".join(c["text"] for c in res.get("content", []) if "text" in c)[:150],
    "joints": joints,
    "moved_frac": float((np.abs(after.astype(int) - before.astype(int)).sum(2) > 8).mean()),
    "sat_frac": float(((after.max(2) - after.min(2)) > 45).mean()),
}
np.save(OUT.with_suffix(".before.npy"), before)
np.save(OUT.with_suffix(".after.npy"), after)
print("honored:", facts["honored"]["status"], facts["honored"]["text"][:60],
      "moved", round(facts["honored"]["moved_frac"], 4), "sat", round(facts["honored"]["sat_frac"], 3))
save()
sim.cleanup()
print("WROTE", OUT)
