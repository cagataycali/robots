"""Render what the horizon pair actually runs. Prints its own tree first."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

import strands_robots.simulation.policy_runner as runner_mod

TREE = Path(runner_mod.__file__).parents[2]
print("TREE:", TREE, flush=True)

from strands_robots import Simulation  # noqa: E402
from strands_robots.policies import Policy  # noqa: E402
from strands_robots.simulation.policy_runner import PolicyRunner  # noqa: E402

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

# A ramp that saturates at RAMP applied actions, so how far the arm has
# travelled IS the horizon the runner honored.
RAMP = 500
TARGETS = {"Rotation": 1.30, "Pitch": -1.50, "Elbow": 1.90, "Jaw": 1.20}


class RampPolicy(Policy):
    """One action per inference; the commanded pose is indexed by the call count."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def provider_name(self) -> str:
        return "ramp"

    def set_robot_state_keys(self, keys: list[str]) -> None:
        self._keys = list(keys)

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, float]]:
        self.calls += 1
        f = min(1.0, self.calls / RAMP)
        return [{k: v * f for k, v in TARGETS.items() if k in self._keys}]


def build() -> Simulation:
    sim = Simulation(backend="mujoco", tool_name="art", mesh=False)
    sim.create_world()
    sim.add_robot(name="so100")
    # Camera BEFORE the rollout: add_camera recompiles the spec and drops ctrl.
    sim.add_camera(name="look", position=[0.62, -0.52, 0.42], target=[0.0, 0.0, 0.16], fov=42)
    return sim


def shoot(sim: Simulation, name: str) -> str:
    r = sim.render(camera_name="look", width=760, height=680)
    assert r.get("status") == "success", r
    png = next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)
    p = OUT / f"{name}.png"
    p.write_bytes(png)
    return str(p)


def scenario(label: str, **kw: Any) -> dict[str, Any]:
    sim = build()
    policy = RampPolicy()
    policy.set_robot_state_keys(sim.robot_action_keys("so100"))
    home = shoot(sim, f"{label}_home")
    try:
        res = PolicyRunner(sim).run(
            "so100", policy, control_frequency=50.0, action_horizon=1, control_substeps=10, **kw
        )
        status, msg = res.get("status"), ""
        blocks = [b for b in res.get("content", []) if "json" in b]
        steps = blocks[0]["json"].get("n_steps") if blocks else None
        reason = blocks[0]["json"].get("stopped_reason") if blocks else None
    except BaseException as exc:  # noqa: BLE001 - the refusal IS the outcome
        status, msg, steps, reason = "raised", str(exc), None, None
    after = shoot(sim, f"{label}_after")
    obs = sim.get_observation(robot_name="so100")
    joints = {k: round(float(v), 4) for k, v in obs.items() if not hasattr(v, "shape")}
    sim.cleanup()
    return {
        "label": label,
        "status": status,
        "msg": msg[:130],
        "steps": steps,
        "stopped_reason": reason,
        "inferences": policy.calls,
        "home_png": home,
        "after_png": after,
        "joints": joints,
    }


facts: dict[str, Any] = {"tree": str(TREE), "ramp": RAMP, "scenarios": {}}
facts["scenarios"]["reference"] = scenario("reference", n_steps=120)
facts["scenarios"]["zero"] = scenario("zero", n_steps=0)

(OUT / "facts.json").write_text(json.dumps(facts, indent=1, default=str))
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "joints"} for k, v in facts["scenarios"].items()}, indent=1))
