"""Capture the frame the gated render assertion verifies, plus the measured tables."""

from __future__ import annotations

import json
import pathlib
import sys

import imageio.v3 as iio
import numpy as np

import strands_robots.simulation.mujoco.simulation as sim_mod

TREE = pathlib.Path(sim_mod.__file__).parents[3]
print("TREE:", TREE)

OUT = pathlib.Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

from strands_robots.simulation import Simulation  # noqa: E402


def _png(result: dict) -> bytes:
    assert result.get("status") == "success", result
    return next(b["image"]["source"]["bytes"] for b in result["content"] if "image" in b)


# Reproduce exactly what the in-scope gated case
# (TestTheSessionSurvives::test_the_world_still_renders_after_a_refused_lookup)
# asserts: seed a world holding one named body, make a refused non-string
# lookup, then assert the world still renders.
sim = Simulation(tool_name="artifact_gl_gate", mesh=False)
facts: dict[str, object] = {"tree": str(TREE)}
try:
    assert sim.create_world()["status"] == "success"
    assert sim.add_object(name="crate", shape="box", size=[0.12, 0.12, 0.12], position=[0.0, 0.0, 0.35])[
        "status"
    ] == "success"
    assert sim.add_camera(name="look", position=[0.52, -0.46, 0.40], target=[0.0, 0.0, 0.18], fov=34)[
        "status"
    ] == "success"
    assert sim.step(n_steps=60)["status"] == "success"

    refused = sim.get_body_state(body_name=None)
    facts["refused_status"] = refused["status"]
    facts["refused_text"] = "".join(b.get("text", "") for b in refused.get("content", []))[:150]

    rendered = sim.render(camera_name="look", width=760, height=560)
    facts["render_status"] = rendered["status"]
    frame = iio.imread(_png(rendered))
    np.save(OUT / "frame.npy", frame)
    facts["frame_shape"] = list(frame.shape)
    sat = float(((frame.max(2).astype(int) - frame.min(2).astype(int)) > 45).mean())
    facts["saturated_frac"] = round(sat, 4)
finally:
    sim.cleanup(policy_stop_timeout=0.5)

assert facts["render_status"] == "success", facts
assert float(facts["saturated_frac"]) > 0.05, facts  # the frame has content

(OUT / "facts.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
print(json.dumps(facts, indent=2))
