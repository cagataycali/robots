# RL Checkpoint (trained actor)

`create_policy("rl", checkpoint_dir=...)` returns an `RLCheckpointPolicy` - the
deterministic actor from a reinforcement-learning run, presented as an ordinary
[`Policy`](overview.md). It is the inference half of
[Reinforcement Learning](../training/rl.md): `create_trainer("ppo")` (or
`fast_sac` / `fast_td3`) writes `policy.pt` + `policy_meta.json`, and this
provider rolls that pair out.

```python
from strands_robots import Robot, create_policy
from strands_robots.training import create_trainer

result = create_trainer("ppo").train(spec)      # -> result.checkpoint_dir

sim = Robot("so101", mode="sim")
sim.run_policy(
    robot_name="so101",
    policy_provider="rl",
    policy_config={"checkpoint_dir": result.checkpoint_dir},
    duration=10.0,
)
```

`checkpoint_dir` is spelled as the trainer spells it (`TrainResult.checkpoint_dir`,
`BaseRLAlgo.load_checkpoint`, `latest_checkpoint`), so the value a caller already
holds is the value this provider takes.

| Config key       | Default | Meaning                                                       |
|------------------|---------|---------------------------------------------------------------|
| `checkpoint_dir` | -       | Directory holding `policy.pt` + `policy_meta.json` (required)  |
| `device`         | `cpu`   | Torch device to load the actor onto                            |

## What the checkpoint decides

`policy_meta.json` is read, not guessed. It carries the vocabularies and the
network shape, and each one is load-bearing:

- **`provider`** selects the actor architecture. The three backends do not share
  one network: PPO's actor emits `num_actions` raw Gaussian means, FastTD3's
  emits `num_actions` through a `tanh`, and FastSAC's emits `2 * num_actions` (a
  mean/log-std pair) and squashes the mean. All three record `num_actions`, so
  the output width is *not* derivable from the metadata alone. The actor is
  rebuilt through the backend's own `build_actor_critic` and driven through its
  own `act_inference`, so the deployed graph is the one that was trained.
- **`actor_obs_keys`** are read from the observation by name, in the trained
  order - that order is part of the weights. A reshuffled observation dict gives
  the same action; a key the observation does not carry is refused, because
  substituting a zero would command the robot from a state it is not in.
- **`action_keys`** name the actuators the outputs drive - the robot's
  `robot_action_keys`, not its joint list. They win over `set_robot_state_keys`,
  which is the fallback for a checkpoint saved without a robot bound. An actor
  whose width does not match the bound keys is refused rather than silently
  dropping a command.

The observation normalizer saved beside the weights is restored in eval mode, so
the statistics the run finished on are frozen: `EmpiricalNormalization` only
folds a batch while training, and a rollout that kept updating would drift the
whitening the trained weights expect.

Actions are one tick per call - an RL actor is a per-step controller trained on
the state it is given, so it has no horizon to predict over. `instruction` is
ignored: the actor was trained against a reward function, not language.

Needs `torch`, which the RL trainers already require; no extra beyond them.
Reading a checkpoint's actor without the policy wrapper is
`strands_robots.training.rl.load_deployable_actor(checkpoint_dir)`.
