---
description: What randomize() actually samples - colors, lighting, physics, positions.
---

# Domain randomization

```python
sim.randomize(
    randomize_colors=True,      # resample object/floor RGB from color_range
    randomize_lighting=True,    # perturb directional + ambient light
    randomize_physics=False,    # mass (mass_range) + friction (friction_range) + damping
    randomize_positions=False,  # add position_noise (m) to every object position
    position_noise=0.02,
    color_range=(0.1, 1.0),
    friction_range=(0.5, 1.5),
    mass_range=(0.5, 2.0),
    seed=42,                    # deterministic sequence
)
```

**Destructive** - writes into MuJoCo model arrays. To restore: `load_scene(...)` or recreate the sim.

`randomize()` leaves the sim in a forwarded, render-ready state: the next `render()` / `get_observation()` reflects the perturbation immediately, with no manual `step()` in between. This matters for lighting in particular - the renderer reads light positions from the derived `data.light_xpos`, not `model.light_pos`, so a light-position jitter only reaches a render after a forward.

## Categories

| Flag | What changes | Range param |
|------|-------------|-------------|
| `randomize_colors` | Object + floor RGB (alpha fixed at 1.0) | `color_range` |
| `randomize_lighting` | Directional direction, intensity, ambient | - |
| `randomize_physics` | Per-object mass (mult), per-geom friction (scale), joint damping | `mass_range`, `friction_range` |
| `randomize_positions` | Object position offsets (metres) | `position_noise` |

Defaults: `colors=True`, `lighting=True`; `physics` and `positions` default `False`.

## Use in an eval loop

```python
for episode in range(N):
    sim.reset()
    sim.randomize(randomize_colors=True, randomize_physics=True, seed=episode)
    # eval_policy has no randomize= kwarg - call sim.randomize() before each episode
    result = sim.eval_policy(robot_name="so100", n_episodes=1, max_steps=300,
                             success_fn=my_fn)
```

## Sensor noise

`set_obs_noise` adds Gaussian measurement noise to observations so a policy is
not trained (or evaluated) on noise-free sensing - a cheap sim-to-real
robustness lever that is orthogonal to `randomize()` (which perturbs the world;
this perturbs the *sensor*).

```python
sim.set_obs_noise(
    joint_pos_std=0.01,      # rad, added to joint positions
    joint_vel_std=0.05,      # rad/s, added to per-joint velocities
    camera_jitter_px=2,      # max integer pixel shift per axis on rendered frames
    seed=0,                  # reproducible noise stream
)
```

Once configured, the noise is applied on every `get_observation`
(joint positions, the `<joint>.vel` entries, and camera frames),
`get_robot_state` (position + velocity), and `render` until reconfigured. Pass
all-zero std to disable; leaving it unconfigured (the default) is an exact
no-op, so existing observations and renders are unchanged. Floating-base
`base_quat` / `base_ang_vel` signals are left untouched (a quaternion would need
renormalization). Values must be finite and non-negative or the call returns
`status=error`.

## Newton backend

The Newton (GPU) backend mirrors both the `randomize` contract for the axes it
supports (colors, lighting, physics) and the `set_obs_noise` sensor-noise
contract, so an identical call behaves the same on either backend. See
[Newton backend](newton.md#domain-randomization-and-sensor-noise).

## See also

- [Simulation overview](overview.md)
- [World building](world-building.md)
- [Recording](../recording.md)
- [Real hardware](../hardware/robot-control.md)
