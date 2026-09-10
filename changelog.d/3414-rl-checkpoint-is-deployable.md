### Added: `create_policy("rl")` - an RL training checkpoint drives a robot

The RL trainers already exported a checkpoint they described as deployable:
`save_checkpoint` writes `policy.pt` + `policy_meta.json` and documents the
latter as "a deployable-policy metadata file", `export()` returns "the loadable
policy artifact for inference", and `act_inference` is docstringed on all three
backends as "the deterministic (mean) action - the deployable policy". Nothing
loaded it. No provider in `strands_robots/registry/policies.json` read an RL
checkpoint, so `create_policy("rl")` raised `Unknown policy provider` and
`docs/training/rl.md`'s "deployable checkpoint whose actor commands ..." had no
path behind it - a policy could be trained in sim and then only evaluated by the
trainer that produced it, never rolled out through `run_policy` / `eval_policy`
like every other provider. That is also why `strands_robots.training.rl` had no
importer anywhere in the package: the loop was open at the deployment end.

`RLCheckpointPolicy` closes it. `create_policy("rl", checkpoint_dir=...)` - the
directory spelling the trainer already uses (`TrainResult.checkpoint_dir`,
`BaseRLAlgo.load_checkpoint`, `latest_checkpoint`) - loads the pair and presents
the trained actor as an ordinary `Policy`, so a PPO / FastSAC / FastTD3 run
deploys with no hand-built glue:

```python
result = create_trainer("ppo").train(spec)
sim.run_policy(robot_name="so101", policy_provider="rl",
               policy_config={"checkpoint_dir": result.checkpoint_dir})
```

The architecture is rebuilt through the backend's own `build_actor_critic`
rather than reimplemented, because the checkpoint metadata does not determine it:
PPO's actor emits `num_actions` raw Gaussian means, FastTD3's emits
`num_actions` through a `tanh`, and FastSAC's emits `2 * num_actions` (a
mean/log-std pair) and squashes the mean, while all three record the same
`num_actions`. `provider` is what selects the graph, and each backend's own
`act_inference` supplies its squash, so a FastSAC actor cannot be loaded into a
PPO-shaped network and silently mis-scaled. The three builders are public for
that reason (`_build_actor_critic` -> `build_actor_critic`, taking the
`hidden_dims` / `init_noise_std` they actually read instead of a whole
`RLTrainSpec`), and the reader lives beside the writer in
`training/rl/checkpoint.py` as `load_deployable_actor`, usable directly for a
custom control loop.

Everything that decides what a command *means* is read rather than assumed.
`actor_obs_keys` are bound by name in the trained order - that order is part of
the weights, so a reshuffled observation dict yields the same action while
swapping two values does not - and a key the observation does not carry is
refused instead of defaulted, because substituting a zero commands the robot
from a state it is not in. The checkpoint's `action_keys` win over
`set_robot_state_keys` (which remains the fallback for a checkpoint saved
without a robot bound), and an actor whose width does not match the bound keys
is refused rather than silently dropping the last command. The saved observation
normalizer is restored in eval mode, so the statistics the run finished on stay
frozen: `EmpiricalNormalization` only folds a batch while training, and a
rollout that kept updating would drift the whitening the trained weights expect.
Missing weights, missing metadata, metadata that is not an object, an omitted
required field and an unknown `provider` each refuse by name.

The provider is also allowlisted for the fleet rail: `_REGISTRY_POLICY_PROVIDERS`
in `strands_robots/mesh/security.py` gates which `policy_provider` a mesh /
Device Connect `execute` or `start` payload may name, and a provider absent from
it is refused on the wire. Widening it does not relax any other gate - the
`policy_host`, `server_address`, `pretrained_name_or_path` and `model_path`
allowlists still apply to every payload.

Documented in `docs/policies/rl.md`, the provider table, and a "Deploying the
checkpoint" section on `docs/training/rl.md`.
