"""Capture the rollout-horizon evidence. Run inside the tree being measured."""

from __future__ import annotations

import json
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np  # noqa: E402

import strands_robots.simulation.base as _b  # noqa: E402
from strands_robots.policies import Policy  # noqa: E402
from strands_robots.simulation import create_simulation  # noqa: E402

TREE = pathlib.Path(_b.__file__).parents[2]
print("TREE:", TREE)

OUT = pathlib.Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
REQUESTED = 120
TARGETS = {
    "Rotation": 1.20,
    "Pitch": -1.60,
    "Elbow": 1.60,
    "Wrist_Pitch": 0.90,
    "Wrist_Roll": 1.20,
    "Jaw": 1.20,
}


class RampPolicy(Policy):
    """Commanded pose is indexed by the CALL COUNT, so travel == honored horizon."""

    def __init__(self, keys, total):
        super().__init__()
        self._keys = list(keys)
        self._total = total
        self.calls = 0

    @property
    def provider_name(self) -> str:
        return "ramp"

    def set_robot_state_keys(self, keys) -> None:  # noqa: D102
        pass

    async def get_actions(self, observation_dict, instruction, **kwargs):  # noqa: D102
        self.calls += 1
        f = min(1.0, self.calls / float(self._total))
        return [{k: TARGETS[k] * f for k in self._keys}]


def joints(sim, name="arm1"):
    obs = sim.get_observation(robot_name=name, skip_images=True)
    return {k: round(float(v), 6) for k, v in sorted(obs.items()) if not hasattr(v, "shape")}


def render(sim, path):
    r = sim.render(camera_name="look", width=560, height=470)
    blob = next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)
    pathlib.Path(path).write_bytes(blob)


def scenario(tag, horizon):
    sim = create_simulation()
    sim.create_world()
    sim.add_robot("arm1", data_config="so100")
    sim.add_camera(name="look", position=[0.52, -0.46, 0.36], target=[0.0, 0.0, 0.13], fov=45)
    keys = sim.robot_action_keys("arm1")
    policy = RampPolicy(keys, REQUESTED)
    render(sim, OUT / f"{tag}_before.png")
    result = sim.run_policy("arm1", policy_object=policy, n_steps=horizon, fast_mode=True)
    text = " ".join(b["text"] for b in result.get("content", []) if "text" in b).replace("\n", " ")
    report = next((b["json"] for b in result["content"] if "json" in b), {})
    render(sim, OUT / f"{tag}_after.png")
    js = joints(sim)
    sim.cleanup()
    return {
        "requested": repr(horizon),
        "status": result["status"],
        "text": text.strip()[:200],
        "steps": report.get("n_steps"),
        "policy_calls": policy.calls,
        "joints": js,
        "travel": round(float(np.sum(np.abs(list(js.values())))), 4),
    }


facts = {
    "tree": str(TREE),
    "requested_reference": REQUESTED,
    "reference": scenario("reference", REQUESTED),
    "bool_horizon": scenario("bool", True),
    "fractional_horizon": scenario("frac", 2.7),
}
(OUT / "facts.json").write_text(json.dumps(facts, indent=2))
print(json.dumps(facts, indent=2)[:1600])
