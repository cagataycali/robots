---
description: When Kimodo samples a motion again, the seed contract, and chaining prompts.
---

# Kimodo sampling and chaining

What decides whether a tick drains the motion buffer or runs the diffusion
sampler again, and how a changed prompt is joined to the pose already commanded.
Installing the policy, its config fields and the checkpoint refusals are
[Kimodo](kimodo.md).

## When the sampler runs again

One `sample()` call produces a motion buffer that `get_actions` drains one frame
per control tick, holding the last frame once exhausted. The buffer is keyed on
the four inputs that determine it — the prompt plus `diffusion_steps`,
`guidance_scale` and `seed` — so the sampler runs again as soon as any of them
differs, and otherwise the buffered frames are reused:

```python
await policy.get_actions({}, "walking forward")                     # samples
await policy.get_actions({}, "walking forward")                     # drains
await policy.get_actions({}, "waving")                              # samples
await policy.get_actions({}, "waving", diffusion_steps=25)          # samples
policy.reset()                                                      # rewinds
policy.reset(seed=7); await policy.get_actions({}, "waving")        # samples
```

Per-call overrides keep the config fields' domains and are checked before the
key is built, so a refused override costs neither a diffusion run nor a frame.
The seed must be a whole number on every surface that sets one. This is
what makes a multi-episode `eval_policy` meaningful — `PolicyRunner.evaluate`
hands each episode its own seed through `reset(seed=...)`, so every episode
samples its own motion and the run replays exactly at the same master `seed=`.

## Chaining prompts into a long-horizon sequence

Because a changed prompt samples the next segment and the stream simply
continues, a long-horizon episode is a rollout that changes the instruction as
it goes; a `policy_object` driven directly is the smallest version:

```python
import asyncio

from strands_robots import Robot
from strands_robots.policies.kimodo import KIMODO_G1_JOINTS, KimodoPolicy

CHAIN = [
    ("a person walking forward with confident strides", 90),
    ("a person turning to the left", 60),
    ("a person waving with the right hand", 60),
    ("a person crouching down to pick an object off the floor", 90),
    ("a person walking forward with confident strides", 90),
]

sim = Robot("g1", mesh=False)
policy = KimodoPolicy()
policy.set_robot_state_keys(list(KIMODO_G1_JOINTS))

for instruction, ticks in CHAIN:
    for _ in range(ticks):
        action = asyncio.run(policy.get_actions({}, instruction))[0]
        sim.set_joint_positions(action, robot_name="g1")
```

Each segment is sampled once, on the tick its instruction first appears. Kimodo
samples every motion from its own canonical start pose, so a new segment is
eased off the pose last commanded across `transition_frames` native frames
(default 5, the sampler's own `num_transition_frames`; minimum 1). Easing
shifts where a segment starts, not how it moves - the root orientation takes the
rotational form of the same offset, so a turn keeps its rate - and removes the
discontinuity without re-planning the motion. An
episode boundary is not a seam: `reset()` forgets the last commanded pose.

## See also

- [Kimodo](kimodo.md) - the policy, its install, its config fields and the
  native-runtime adapter.
- [ProtoMotions](protomotions.md) - the tracker that follows a sampled clip under
  physics.
