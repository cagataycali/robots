"""Capture what each max_grad_norm does to a real seeded PPO run, on THIS tree."""
import json, os, pathlib, sys, tempfile
import torch
import strands_robots.training.rl.ppo as PPO
TREE = str(pathlib.Path(PPO.__file__).parents[3])
print("TREE:", TREE, flush=True)

import strands_robots as sr
from strands_robots.training import create_trainer
from strands_robots.training.rl import RLTrainSpec, SimEnv

def _elbow_reward(engine):
    return -abs(float(engine.get_observation(robot_name="so100")["Elbow"]) - 1.0)

def _env():
    return SimEnv(sr.Robot("so100", mode="sim"), actor_obs_keys=["Elbow", "Elbow.vel"],
                  reward_terms=[_elbow_reward], action_dim=6, max_episode_steps=10)

def _flat(sd):
    return torch.cat([v.detach().reshape(-1).double() for k, v in sorted(sd.items())
                      if isinstance(v, torch.Tensor) and v.is_floating_point()])

def run(mgn, freeze=False):
    if freeze:
        PPO.PpoTrainer.update = lambda self: {}
    trainer = create_trainer("ppo")
    with tempfile.TemporaryDirectory() as td:
        spec = RLTrainSpec(env_factory=_env, output_dir=td, total_timesteps=60, rollout_steps=20,
                           num_mini_batches=4, num_learning_epochs=2, seed=0, max_grad_norm=mgn)
        rec = {"value": repr(mgn), "problems": trainer.validate(spec)}
        try:
            res = trainer.train(spec)
        except Exception as exc:
            rec.update(status="RAISED", detail=f"{type(exc).__name__}: {str(exc)[:110]}")
            return rec, None
        rec["status"] = res.status
        rec["detail"] = (res.message or "")[:150]
        if res.checkpoint_dir:
            w = _flat(torch.load(os.path.join(res.checkpoint_dir, "policy.pt"),
                                 weights_only=False)["actor_critic"])
            rec["w_sum"] = f"{float(w.sum()):.10f}"
            return rec, w
        return rec, None

base, w0 = run(1.0, freeze=True)
import importlib; importlib.reload(PPO)
out = {"tree": TREE, "baseline_w_sum": base["w_sum"], "rows": []}

deltas = {}
for c in [1.0, float("inf"), 0.0, -1.0, True, float("nan")]:
    r, w = run(c)
    if w is not None and w0 is not None:
        d = w - w0
        r["delta_norm"] = f"{float(d.norm()):.10f}"
        r["identical_to_untrained"] = bool(torch.equal(w, w0))
        deltas[repr(c)] = d
    out["rows"].append(r)
    print(json.dumps(r), flush=True)

if "1.0" in deltas and "-1.0" in deltas:
    dh, di = deltas["1.0"], deltas["-1.0"]
    out["cosine_honoured_vs_negated"] = f"{float(torch.dot(dh, di) / (dh.norm() * di.norm())):.6f}"

# the consumer's own behaviour, on a gradient of norm 5
from torch import nn
grads = {}
for v in (1.0, float("inf"), 0.0, -1.0, float("nan")):
    p = nn.Parameter(torch.tensor([3.0, 4.0])); p.grad = torch.tensor([3.0, 4.0])
    nn.utils.clip_grad_norm_([p], v)
    grads[repr(v)] = [round(x, 4) for x in p.grad.tolist()]
out["clip_grad_norm_effect"] = grads
pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=str))
print("COSINE:", out.get("cosine_honoured_vs_negated"), flush=True)
