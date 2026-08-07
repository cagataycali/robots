"""Measure the GAE trace-decay domain. Run unchanged in two trees."""
from __future__ import annotations
import json, os, pathlib, sys, tempfile

import strands_robots.training._validate as _v
TREE = str(pathlib.Path(_v.__file__).parents[2])
print("TREE:", TREE)

import numpy as np
import torch
from strands_robots.training import create_trainer
from strands_robots.training.rl import RLTrainSpec
from strands_robots.training.rl.ppo import compute_gae

GAMMA = 0.99
HORIZONS = [6, 12, 24, 48, 96]
LAMS = [0.0, 0.95, 1.0, 1.5, 2.0, -0.5, -2.0, 1e6]


def largest_adv(gamma, lam, T):
    z = torch.zeros(T)
    adv, _ = compute_gae(torch.ones(T), z, z, z, z, gamma, lam)
    return float(adv.abs().max())


# 1) the divergence measurement (a property of the recursion; identical on both trees)
divergence = {repr(l): [largest_adv(GAMMA, l, T) for T in HORIZONS] for l in LAMS}

# 2) the verdict table through the real validate() entry point
PROBE = [0.0, 0.95, 1.0, 1.5, 2.0, -0.5, -2.0, 1e6, True, float("nan"), float("inf"), "0.95", None]


def verdict(lam):
    spec = RLTrainSpec(output_dir="/tmp/art_lam", env_factory=lambda: None)
    spec.lam = lam
    try:
        problems = [p for p in create_trainer("ppo").validate(spec) if p.startswith("ppo: lam ")]
    except Exception as e:  # noqa: BLE001 - an escape past the documented list return is an answer
        return {"outcome": "raised", "text": f"{type(e).__name__}: {e}"}
    if problems:
        return {"outcome": "refused", "text": problems[0]}
    return {"outcome": "accepted", "text": ""}


verdicts = {repr(l): verdict(l) for l in PROBE}

# fast_sac must stay quiet: it has no advantage trace
sac_quiet = {}
for l in [1.5, float("nan")]:
    spec = RLTrainSpec(output_dir="/tmp/art_lam", env_factory=lambda: None)
    spec.lam = l
    sac_quiet[repr(l)] = [p for p in create_trainer("fast_sac").validate(spec) if p.startswith("fast_sac: lam ")]

# 3) a real PPO run on an honored spec: the no-regression proof
def _elbow_reward(engine):
    return -abs(float(engine.get_observation(skip_images=True)["Elbow"]) - 0.2)


def _make_env():
    import strands_robots as sr
    from strands_robots.training.rl import SimEnv

    return SimEnv(sr.Robot("so100", mode="sim"), actor_obs_keys=["Elbow", "Elbow.vel"],
                  reward_terms=[_elbow_reward], action_dim=6, max_episode_steps=10)


run = {}
with tempfile.TemporaryDirectory() as td:
    spec = RLTrainSpec(env_factory=_make_env, output_dir=td, total_timesteps=20 * 3,
                       rollout_steps=20, num_mini_batches=4, num_learning_epochs=2, seed=0)
    problems = create_trainer("ppo").validate(spec)
    res = create_trainer("ppo").train(spec)
    run["validate_problems"] = problems
    run["status"] = res.status
    run["lam"] = spec.lam
    ckpt = os.path.join(res.checkpoint_dir or "", "policy.pt")
    if os.path.isfile(ckpt):
        sd = torch.jit.load(ckpt).state_dict() if ckpt.endswith(".pt") and False else None
        obj = torch.load(ckpt, map_location="cpu", weights_only=False)
        flat = []
        def walk(o):
            if isinstance(o, torch.Tensor):
                flat.append(o.detach().float().flatten())
            elif isinstance(o, dict):
                for k in sorted(o): walk(o[k])
            elif isinstance(o, (list, tuple)):
                for x in o: walk(x)
            elif hasattr(o, "state_dict"):
                walk(o.state_dict())
        walk(obj)
        if flat:
            w = torch.cat(flat)
            run["n_params"] = int(w.numel())
            run["w_absmax"] = f"{float(w.abs().max()):.16f}"
            run["w_sum"] = f"{float(w.sum()):.16f}"

out = {"tree": TREE, "gamma": GAMMA, "horizons": HORIZONS, "lams": [repr(l) for l in LAMS],
       "divergence": divergence, "probe": [repr(l) for l in PROBE], "verdicts": verdicts,
       "sac_quiet": sac_quiet, "run": run}
pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
print("wrote", sys.argv[1], "| run:", run.get("status"), "| w_absmax:", run.get("w_absmax"))
