"""Capture: preflight verdicts + a control training run, on whichever tree runs it."""
import json, math, pathlib, sys, tempfile
import numpy as np, torch
import strands_robots
TREE = str(pathlib.Path(strands_robots.__file__).parents[1])
print("TREE:", TREE, flush=True)
from strands_robots.training import create_trainer
from strands_robots.training.rl import RLTrainSpec, SimEnv
import strands_robots as sr

PROBES = [("nan", math.nan), ("inf", math.inf), ("-inf", -math.inf), ("True", True),
          ("'1.0'", "1.0"), ("None", None), ("[1.0]", [1.0]),
          ("0.0", 0.0), ("-0.5", -0.5), ("1.0", 1.0)]
FIELDS = ["value_loss_coef", "entropy_coef"]

def verdict(field, value):
    spec = RLTrainSpec(output_dir="/tmp/verdict", env_factory=lambda: None)
    setattr(spec, field, value)
    probs = [p for p in create_trainer("ppo").validate(spec) if p.startswith(f"ppo: {field} ")]
    return "refused" if probs else "accepted"

grid = {f: {n: verdict(f, v) for n, v in PROBES} for f in FIELDS}

def _elbow_reward(e):
    return -abs(float(e.get_observation(robot_name="so100")["Elbow"]) - 1.0)
def _env():
    return SimEnv(sr.Robot("so100", mode="sim"), actor_obs_keys=["Elbow", "Elbow.vel"],
                  reward_terms=[_elbow_reward], action_dim=6, max_episode_steps=10)
def fingerprint(path):
    ck = torch.load(path, weights_only=False); flat = []
    def walk(o):
        if isinstance(o, torch.Tensor): flat.append(o.detach().double().cpu().reshape(-1))
        elif isinstance(o, dict):
            for v in o.values(): walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o: walk(v)
        elif hasattr(o, "state_dict"): walk(o.state_dict())
    walk(ck); t = torch.cat(flat)
    return {"n": int(t.numel()), "nan": int(torch.isnan(t).sum()),
            "absmax": f"{float(t.abs().max()):.16f}", "sum": f"{float(t.sum()):.16f}"}

d = tempfile.mkdtemp(prefix="ctl_")
spec = RLTrainSpec(env_factory=_env, output_dir=d, total_timesteps=60, rollout_steps=20,
                   num_envs=1, num_mini_batches=4, num_learning_epochs=2, seed=0,
                   save_freq=1000, device="cpu")
r = create_trainer("ppo").train(spec)
cks = sorted(pathlib.Path(d).rglob("*.pt"))
control = {"status": r.status, "checkpoints": len(cks), "fingerprint": fingerprint(cks[-1]) if cks else None}
print("CONTROL:", json.dumps(control), flush=True)
pathlib.Path(sys.argv[1]).write_text(json.dumps({"tree": TREE, "grid": grid, "control": control}, indent=1))
