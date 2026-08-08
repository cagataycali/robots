"""Measured verdicts + consequences for the optimization-epoch domain."""
from __future__ import annotations
import json, os, pathlib, sys, tempfile
import strands_robots.training.rl.ppo as ppo_mod
TREE = str(pathlib.Path(ppo_mod.__file__).parents[3])
print("TREE:", TREE, flush=True)
from strands_robots.training import create_trainer                     # noqa: E402
from strands_robots.training.rl import RLTrainSpec                     # noqa: E402
from strands_robots.training.rl.ppo import PpoTrainer                  # noqa: E402


def _elbow_reward(engine):
    return -abs(float(engine.get_observation(robot_name="so100")["Elbow"]) - 1.0)


def _make_env():
    import strands_robots as sr
    from strands_robots.training.rl import SimEnv
    return SimEnv(sr.Robot("so100", mode="sim"), actor_obs_keys=["Elbow", "Elbow.vel"],
                  reward_terms=[_elbow_reward], action_dim=6, max_episode_steps=10)


def _fp(path):
    import torch
    obj = torch.load(path, weights_only=False, map_location="cpu")
    flat = []
    def walk(o):
        if isinstance(o, torch.Tensor): flat.append(o.detach().reshape(-1).double())
        elif isinstance(o, dict):
            for v in o.values(): walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o: walk(v)
        elif hasattr(o, "state_dict"): walk(o.state_dict())
    walk(obj["actor_critic"] if isinstance(obj, dict) and "actor_critic" in obj else obj)
    cat = torch.cat(flat)
    return {"n": int(cat.numel()), "sum": f"{float(cat.sum()):.16f}"}


def _spec(epochs, out):
    s = RLTrainSpec(env_factory=_make_env, output_dir=out, total_timesteps=60,
                    rollout_steps=20, num_envs=1, num_mini_batches=4, seed=0)
    s.num_learning_epochs = epochs
    return s


def _run(epochs, *, never_update=False):
    """Real 60-step PPO run; returns verdict + measured consequences."""
    import torch
    out = tempfile.mkdtemp(prefix="ppo-")
    tr, spec = PpoTrainer(), _spec(epochs, out)
    try:
        refused = bool([p for p in tr.validate(spec) if p.startswith("ppo: num_learning_epochs ")])
    except BaseException as e:                                # noqa: BLE001 - probe classifier
        return {"refused": False, "status": f"validate raised {type(e).__name__}", "steps": 0}
    if refused:
        (msg,) = [p for p in tr.validate(spec) if p.startswith("ppo: num_learning_epochs ")]
        return {"refused": True, "status": "refused by preflight", "steps": 0, "message": msg}
    n = {"v": 0}
    real_step, real_upd = torch.optim.Adam.step, PpoTrainer.update
    def counting(self, *a, **k):
        n["v"] += 1
        return real_step(self, *a, **k)
    torch.optim.Adam.step = counting                          # type: ignore[method-assign]
    if never_update:
        PpoTrainer.update = lambda self: {}                   # type: ignore[method-assign,assignment]
    try:
        r = tr.train(spec)
        st, ms = r.status, {k: round(v, 6) for k, v in (r.metrics or {}).items()
                            if k in ("surrogate_loss", "value_loss", "entropy")}
    except BaseException as e:                                # noqa: BLE001 - probe classifier
        st, ms = f"{type(e).__name__}", None
    finally:
        torch.optim.Adam.step = real_step                     # type: ignore[method-assign]
        PpoTrainer.update = real_upd                          # type: ignore[method-assign]
    ck = os.path.join(out, "checkpoints", "last", "policy.pt")
    return {"refused": False, "status": st, "steps": n["v"], "losses": ms,
            "fp": _fp(ck) if os.path.exists(ck) else None}


CASES = [("5 (default)", 5), ("0", 0), ("-3", -3), ("True", True), ("2.7", 2.7), ("None", None)]
res = {"tree": TREE, "cases": {}}
for label, v in CASES:
    res["cases"][label] = _run(v)
    print(f"  {label:12s} {res['cases'][label]}", flush=True)
# the "never trained" control: 5 epochs but update() replaced by a no-op
res["never_trained"] = _run(5, never_update=True)
print(f"  {'NEVER-TRAINED':12s} {res['never_trained']}", flush=True)
pathlib.Path(sys.argv[1]).write_text(json.dumps(res, indent=2))
