import json, math, pathlib, sys
import numpy as np
import strands_robots.simulation.policy_runner as prmod
TREE = str(pathlib.Path(prmod.__file__).parents[2])
print("TREE:", TREE)
from strands_robots import Simulation
from strands_robots.policies import MockPolicy
from strands_robots.simulation.policy_runner import PolicyRunner

OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
ROBOT, STEPS, W, H = "so100", 96, 760, 620

def png(sim, tag):
    r = sim.render(camera_name="look", width=W, height=H)
    assert r.get("status") == "success", r
    data = next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)
    (OUT / f"{tag}.png").write_bytes(data)
    return f"{tag}.png"

def joints(sim):
    obs = sim.get_observation(robot_name=ROBOT)
    return {k: round(float(v), 6) for k, v in sorted(obs.items()) if not hasattr(v, "shape")}

def case(tag, horizon):
    sim = Simulation(backend="mujoco", tool_name=f"art_{tag}", mesh=False)
    sim.create_world()
    sim.add_robot(name=ROBOT)
    sim.add_camera(name="look", position=[0.62, -0.52, 0.42], target=[0.0, 0.0, 0.16], fov=42)
    policy = MockPolicy()
    policy.set_robot_state_keys(sim.robot_action_keys(ROBOT))
    calls = {"n": 0}
    inner = policy.get_actions
    async def counting(*a, **k):
        calls["n"] += 1
        return await inner(*a, **k)
    policy.get_actions = counting  # type: ignore[method-assign]
    rec: dict = {"tag": tag, "horizon": repr(horizon), "tree": TREE}
    try:
        res = PolicyRunner(sim).run(ROBOT, policy, n_steps=STEPS, action_horizon=horizon, fast_mode=True)
        rec["outcome"] = res.get("status")
        rec["text"] = " ".join(str(b.get("text", "")) for b in res.get("content", []))[:150]
    except ValueError as e:
        rec["outcome"] = "raised ValueError"
        rec["text"] = str(e)[:150]
    rec["inferences"] = calls["n"]
    rec["joints"] = joints(sim)
    if rec["outcome"] == "success":
        rec["png"] = png(sim, tag)
    sim.cleanup()
    return rec

rows = [case("honored", 8), case("clamped", 0), case("nonfinite", math.nan)]
(OUT / "facts.json").write_text(json.dumps({"tree": TREE, "steps": STEPS, "rows": rows}, indent=2))
for r in rows:
    print(f"  {r['tag']:<10} h={r['horizon']:<5} {r['outcome']:<18} infers={r['inferences']:<4} {r['text'][:80]}")
