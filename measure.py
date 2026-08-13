"""Measure the policy-running guard across the three Isaac primitives."""
import json, pathlib, sys
import strands_robots
print("TREE:", pathlib.Path(strands_robots.__file__).parents[1])
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import types
from tests.simulation.isaac.test_motion_primitives import (
    _FakeArticulation, _FakeArticulationAction, _FakeWorld, _make_sim, ARM_JOINTS, ARM_LIMITS,
)

# install the fake isaacsim types (the autouse fixture does this in pytest)
for name in ("isaacsim", "isaacsim.core", "isaacsim.core.utils", "isaacsim.core.utils.types"):
    sys.modules[name] = types.ModuleType(name)
sys.modules["isaacsim.core.utils.types"].ArticulationAction = _FakeArticulationAction

rows = []

# --- up-front: policy already running -------------------------------------
for prim, kwargs in (
    ("move_to", dict(position=[0.2, 0.0, 0.2])),
    ("set_gripper", dict(state="open")),
    ("rotate_wrist", dict(target_yaw=0.4)),
):
    sim, art = _make_sim()
    sim._robots["arm"].policy_running = True
    res = getattr(sim, prim)(robot_name="arm", **kwargs)
    rows.append({
        "phase": "already-running",
        "primitive": prim,
        "status": res["status"],
        "text": res["content"][0].get("text", "")[:200],
        "actions_applied": len(art.applied),
        "names_policy": "policy" in res["content"][0].get("text", "").lower(),
    })

# --- mid-run: policy starts while the primitive drives --------------------
for prim, kwargs in (
    ("set_gripper", dict(state="open", steps=50)),
    ("rotate_wrist", dict(target_yaw=0.7, max_steps=50)),
):
    box = {}
    def _start_policy():
        box["sim"]._robots["arm"].policy_running = True
    sim, art = _make_sim(servo_rate=0.0, on_step=_start_policy)
    box["sim"] = sim
    res = getattr(sim, prim)(robot_name="arm", **kwargs)
    rows.append({
        "phase": "starts-mid-run",
        "primitive": prim,
        "status": res["status"],
        "text": res["content"][0].get("text", "")[:200],
        "actions_applied": len(art.applied),
        "ticks": sim._world.steps,
        "names_policy": "policy" in res["content"][0].get("text", "").lower(),
    })

for r in rows:
    print(json.dumps(r))
_root = pathlib.Path(strands_robots.__file__).parents[1]
pathlib.Path("/tmp/matrix-%s.json" % _root.name).write_text(json.dumps({"tree": str(_root), "rows": rows}, indent=2))
print("WROTE /tmp/matrix-%s.json" % _root.name)
