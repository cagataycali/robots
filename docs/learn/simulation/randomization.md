---
description: Domain randomization and sensor noise on every backend, when to call them, and the sim to real habits the rest of the package supports.
---

# Randomization

By the end of this page you can randomize colours, lights, friction, mass and object positions with one seeded call, add encoder and camera noise to every observation, and know the ordering rule that decides whether a rollout sees the randomized scene at all.

```python
from strands_robots.simulation import create_simulation

sim = create_simulation("mujoco")
sim.create_world()
sim.add_robot("so101")
sim.add_object(name="cube", shape="box", size=[0.03, 0.03, 0.03], position=[0.25, 0.0, 0.015])
clean = sim.get_observation("so101", skip_images=True)["1"]

r = sim.randomize(randomize_colors=True, randomize_lighting=True, randomize_physics=True, randomize_positions=True,
                  position_noise=0.02, friction_range=(0.7, 1.3), mass_range=(0.8, 1.2), seed=7)
print("\n".join(r["content"][0]["text"].splitlines()[:4]))
print(r["content"][0]["text"].splitlines()[-1])

sim.set_obs_noise(joint_pos_std=0.002, joint_vel_std=0.01, camera_jitter_px=1.0, seed=7)
noisy = sim.get_observation("so101", skip_images=True)["1"]
print(clean, noisy != clean, abs(noisy - clean) < 0.01)
sim.set_obs_noise()
print(sim.get_observation("so101", skip_images=True)["1"] == clean)
sim.cleanup()
```

You should see:

```text
Domain Randomization applied:
Colors: 31 geoms randomized
Lighting: 2 lights randomized
Physics: 32 geoms friction-scaled, 8 bodies mass-scaled
Positions: 1 dynamic objects perturbed by +/-0.02m
0.0 True True
True
```

The full text also lists every friction and mass scale by geom and body name, so a run is reproducible from its log.

## randomize

`randomize(randomize_colors=True, randomize_lighting=True, randomize_physics=False, randomize_positions=False, position_noise=0.02, color_range=(0.1, 1.0), friction_range=(0.5, 1.5), mass_range=(0.5, 2.0), seed=None)`. Each flag is one axis:

| axis | what changes |
|---|---|
| `randomize_colors` | every non-ground geom's RGB and its material colour, sampled in `color_range` |
| `randomize_lighting` | each light's position inside 0.5 m of its authored spot, and its diffuse colour |
| `randomize_physics` | every geom's friction scaled in `friction_range`, every body's mass in `mass_range` |
| `randomize_positions` | every dynamic object's position perturbed by `position_noise` metres, written to `qpos0` too |

The flags are strict booleans: `"false"`, `"no"`, `"off"` and `"0"` are truthy strings and are refused rather than turning an axis on. A keyword the call does not honour (`randomize_position`, `position_range`) is refused with the valid set; both methods declare `**kwargs` only to match the base signature and forward nothing. With every flag off the call is a no-op. `seed` makes the draw deterministic.

## When to call it

Randomization writes the compiled model and survives `reset()`, which is why it reaches a rollout: `run_policy` and `eval_policy` reset before an episode's first step. It does not survive a scene mutation. `add_object`, `remove_object`, `add_camera`, `remove_camera`, `add_robot`, `remove_robot` and `patch_scene_mjcf` rebuild the model from the authored spec and restore every value. Randomizing before one of them is a silent no-op: both calls report success and the policy's first observation is the authored scene. Build the scene, then randomize, then roll out.

## set_obs_noise

`set_obs_noise(joint_pos_std=0.0, joint_vel_std=0.0, camera_jitter_px=0.0, seed=None)` adds Gaussian noise to every joint reading and jitters every rendered frame by up to the given pixels, on `get_observation`, `get_robot_state` and `render`, until reconfigured. All-zero standard deviations are an exact no-op, so an unconfigured engine returns observations byte for byte unchanged. MuJoCo, Newton and Isaac share one implementation (`ObservationNoiseMixin`), so one call behaves the same on each.

## From randomization to sim to real

The package's habits for a policy that has to survive the transfer, each on its own page:

| habit | where |
|---|---|
| randomize physics and sensors per episode, seeded, after the scene is built | this page |
| hold the state vector to the real driver's key order and units (`.pos` real entries, `dim_policy`) | [lerobot-local](../policies/lerobot-local.md) |
| name sim cameras after the embodiment's source keys so the same policy config runs on hardware | [lerobot-local](../policies/lerobot-local.md) |
| step physics for the full control period, not one `dt`, so a position servo tracks each target | [simulation](index.md) |
| read the `[sim] action value ... outside the range` warning as a units mismatch: the value is written verbatim to `ctrl`, the joint cannot follow it, and the call still reports success | [MuJoCo](mujoco.md) |
| record sim episodes into the same LeRobot dataset format the real recorder writes | [data](../data/record.md) |
| evaluate with `pass_hat_k`, not the mean success rate, before a deployment decision | [predicates and rollouts](predicates-and-rollouts.md) |

`actuate_robot(robot_name, kp=100.0, damping=2.0, armature=0.01, gravity_compensation=True)` on MuJoCo turns an actuator-less URDF import into a position-servo arm; tune `kp` and `damping` toward the real controller before trusting a transfer.
