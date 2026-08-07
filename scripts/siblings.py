"""Measure the other bare-coerced numerics of the same SimEnv signature."""
from __future__ import annotations
import json, pathlib
import strands_robots.training.rl.env as envmod
print("TREE:", pathlib.Path(envmod.__file__).parents[3], flush=True)
import torch
from typing import cast, Any
from strands_robots.simulation.base import SimEngine

class _Fake:
    def list_robots(self): return ["fake"]
    def robot_joint_names(self, n): return ["A", "B"]
    def robot_action_keys(self, n): return ["A", "B"]
    def reset(self): return {"status": "success"}
    def get_observation(self, robot_name=None, *, skip_images=False): return {"A": 0.0, "B": 0.0}
    def send_action(self, action, robot_name=None, n_substeps=1): return {"status": "success"}

from strands_robots.training.rl import SimEnv

def probe(param: str, values: list) -> None:
    print(f"\n### {param}")
    print(f"{'value':<14} {'ctor':<9} {'stored':<10} {'episode behaviour'}")
    print("-" * 78)
    for v in values:
        kw: dict[str, Any] = {"action_dim": 2}
        kw[param] = v
        if param == "action_dim": kw = {"action_dim": v}
        try:
            env = SimEnv(cast(SimEngine, _Fake()), actor_obs_keys=["A"],
                         reward_terms=[lambda e: 1.0], **kw)
        except BaseException as e:  # noqa: BLE001
            print(f"{v!r:<14} RAISED    {type(e).__name__}: {str(e)[:52]}")
            continue
        stored = repr(getattr(env, "max_episode_steps" if param == "max_episode_steps" else "num_actions"))
        note = ""
        if param == "max_episode_steps":
            env.reset()
            outs = []
            for _ in range(3):
                _o, _r, d, info = env.step(torch.zeros(2))
                outs.append((bool(d.item()), info["time_out"]))
            note = f"first 3 steps done/time_out = {outs}"
        else:
            note = f"num_actions={env.num_actions}"
            try:
                env.reset(); env.step(torch.zeros(max(env.num_actions, 1)))
                note += " ; step ok"
            except BaseException as e:  # noqa: BLE001
                note += f" ; step {type(e).__name__}: {str(e)[:30]}"
        print(f"{v!r:<14} accepted  {stored:<10} {note}")

probe("max_episode_steps", [200, 1, 0, -5, True, 2.7, float("nan"), "10", None])
probe("action_dim", [2, 0, -1, True, 2.7, "2", float("nan")])
