"""Capture the benchmark-declaration outcome on whichever tree this runs in."""
from __future__ import annotations
import json, pathlib, sys
import numpy as np
import strands_robots.simulation.base as sb
TREE = str(pathlib.Path(sb.__file__).parents[2])
print("TREE:", TREE)

from strands_robots import Simulation
from strands_robots.simulation.benchmark import register_benchmark, unregister_benchmark
from strands_robots.simulation.benchmark_spec import DeclarativeBenchmark

OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
CAM = dict(name="look", position=[1.35, -1.35, 0.95], target=[0.0, 0.0, 0.42], fov=32)


def render(sim):
    r = sim.render(camera_name="look", width=760, height=680)
    png = next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)
    return png


def evaluate(label, supported, steps):
    """Declare a benchmark, register it, evaluate it. Report every real outcome."""
    row = {"label": label}
    try:
        bench = DeclarativeBenchmark(
            name=label, supported_robots=supported, default_robot="panda", max_steps=steps,
            success_fn=lambda s: False, failure_fn=lambda s: False, reward_terms=[])
    except ValueError as exc:
        row["construction"] = "REFUSED"
        row["message"] = str(exc)
        row["declared"] = None
    else:
        row["construction"] = "accepted"
        row["message"] = ""
        row["declared"] = list(bench.supported_robots)
        unregister_benchmark(label)
        register_benchmark(label, bench)

    sim = Simulation(backend="mujoco", mesh=False)
    try:
        sim.create_world()
        sim.add_robot(name="panda", data_config="panda")
        sim.add_camera(**CAM)                     # before any rollout: add_camera recompiles
        row["png_home"] = render(sim)
        if row["construction"] == "accepted":
            lb = " ".join(c.get("text", "") for c in sim.list_benchmarks()["content"] if "text" in c)
            row["list_benchmarks"] = next(
                (ln.strip() for ln in lb.splitlines() if label in ln), "")
            res = sim.evaluate_benchmark(benchmark_name=label, robot_name="panda",
                                         policy_provider="mock", n_episodes=1)
            row["eval_status"] = res.get("status")
            row["eval_text"] = " ".join(
                c.get("text", "") for c in res.get("content", []) if "text" in c)
        else:
            row["list_benchmarks"] = "(never registered)"
            row["eval_status"] = "(not reached)"
            row["eval_text"] = "(not reached - construction refused)"
        row["png_after"] = render(sim)
        obs = sim.get_observation(robot_name="panda")
        row["joints"] = {k: round(float(v), 6) for k, v in sorted(obs.items())
                         if not hasattr(v, "shape") and k.startswith("joint")}
    finally:
        sim.cleanup()
    return row


rows = [evaluate("declared_as_list", ["panda"], 120),
        evaluate("declared_as_bare_string", "panda", 120)]

facts = {"tree": TREE, "rows": []}
for i, row in enumerate(rows):
    for key in ("png_home", "png_after"):
        p = OUT / f"{row['label']}_{key}.png"
        p.write_bytes(row.pop(key))
        row[key] = str(p)
    facts["rows"].append(row)
(OUT / "facts.json").write_text(json.dumps(facts, indent=2))
for row in facts["rows"]:
    print(f"  {row['label']:24s} ctor={row['construction']:9s} declared={row['declared']} "
          f"eval={row['eval_status']}")
    if row["message"]:
        print(f"      msg: {row['message'][:110]}")
    print(f"      eval: {row['eval_text'][:150]}")
print("OK")
