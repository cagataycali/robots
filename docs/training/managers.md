# Sim Managers: config-driven RL environments

`strands_robots.sim_managers` is a backend-agnostic, config-driven composition
layer for reinforcement-learning environments, modelled on the *manager* pattern
used by Isaac Lab / RSL-RL. Instead of hand-wiring observation, reward,
termination, and command logic inline for every task, you compose them from
atomic, reusable **terms** described declaratively in a single config.

```python
from strands_robots.sim_managers import RewardManager, ObservationManager

reward = RewardManager.from_config({
    "terms": [
        {"name": "track", "func": "track_lin_vel_xy_exp", "weight": 1.0,
         "params": {"std": 0.25}},
        {"name": "z_vel", "func": "lin_vel_z_l2", "weight": -2.0},
    ]
})
r = reward.compute(state)        # scalar reward
reward.term_values               # {"track": ..., "z_vel": ...}
```

## Why managers

Without an abstraction, every example re-implements observation assembly, reward
math, and termination checks. That blocks curriculum learning, multi-task
training (locomotion and whole-body tracking share most terms), and reuse. The
manager pattern makes the whole RL recipe a YAML file: change a weight, add a
penalty, or swap a command source without touching Python.

## Core concepts

### EnvState - the backend-agnostic contract

Every term reads from one object: `EnvState`. A simulator backend (or a rollout
driver) populates it each control step; terms never touch the simulator, so the
*same* recipe runs on MuJoCo, MJWarp, Isaac Gym, Isaac Sim, or Newton.

```python
from strands_robots.sim_managers import EnvState
import numpy as np

state = EnvState(
    joint_pos=np.zeros(12), joint_vel=np.zeros(12),
    action=np.zeros(12), last_action=np.zeros(12),
    base_lin_vel=np.array([0.5, 0.0, 0.0]),     # base frame
    base_ang_vel=np.array([0.0, 0.0, 0.1]),     # base frame
    projected_gravity=np.array([0.0, 0.0, -1.0]),
    base_height=0.78, dt=0.02, step_count=0, max_episode_length=1000,
)
```

Velocities are expressed in the robot's **base frame**; `projected_gravity` is
the gravity unit vector rotated into the base frame (`[0, 0, -1]` upright).
Optional arrays (`joint_torque`, `joint_acc`, `default_joint_pos`) default to
zeros sized to the joint count. See the `EnvState` docstring for the full field
reference. A term that needs a field which was not populated raises a clear
error rather than silently degrading.

### Term - one atomic computation

A `Term` subclass implements `__call__(state) -> value`. Terms are registered in
a closed registry keyed by `(category, func_name)` via `@register_term`. Configs
reference a term only by its `func` name; an unknown name is rejected (no
`eval`), so configs are safe to parse from untrusted / LLM-authored YAML.

```python
from strands_robots.sim_managers import register_term, Term

@register_term("reward", "my_penalty")
class MyPenalty(Term):
    def __init__(self, scale: float = 1.0, **params):
        super().__init__(scale=scale, **params)
        self.scale = scale
    def __call__(self, state):
        return self.scale * float((state.action ** 2).sum())
```

### Manager - combine terms into a result

| Manager | `compute(state)` returns | Combination |
|---|---|---|
| `ObservationManager` | 1-D `float64` vector | scale -> clip -> concatenate (order preserved; `term_slices` maps label -> slice) |
| `RewardManager` | scalar reward | `sum(weight * term(state) * dt)`; `term_values` holds the per-term breakdown |
| `TerminationManager` | `TerminationResult` | classifies `time_out` (truncation) vs `terminated` (failure) |
| `CommandManager` | `dict[str, vector]` | samples + resamples commands, writes them onto `state.commands` |

The reward weight's **sign** distinguishes a reward (positive) from a penalty
(negative), following the Isaac Lab convention. Reward contributions are scaled
by `dt` by default so a recipe is invariant to control frequency
(`scale_by_dt=False` to disable).

## Config DSL

A managers config is a mapping of manager keys to `{"terms": [...]}` blocks. Each
term entry has:

| Key | Applies to | Meaning |
|---|---|---|
| `func` | all | registered term name (required) |
| `name` | all | instance label (defaults to `func`); must be unique per manager |
| `params` | all | keyword arguments forwarded to the term constructor |
| `weight` | reward | term weight (sign = reward vs penalty) |
| `scale` | observation | multiplier applied before clipping |
| `clip` | observation | `[low, high]` applied after scaling |

```yaml
command_manager:
  terms:
    - name: base_velocity
      func: uniform_velocity
      params: {lin_vel_x: [-1.0, 1.0], ang_vel_z: [-1.0, 1.0], resampling_time: 5.0}
observation_manager:
  terms:
    - {func: base_lin_vel, scale: 2.0}
    - {func: velocity_commands}
    - {func: joint_pos}
reward_manager:
  terms:
    - {name: track_lin, func: track_lin_vel_xy_exp, weight: 1.0, params: {std: 0.25}}
    - {name: z_vel, func: lin_vel_z_l2, weight: -2.0}
termination_manager:
  terms:
    - {func: time_out}
    - {func: bad_orientation, params: {limit_angle: 1.0}}
```

Build the whole set at once:

```python
from strands_robots.sim_managers import build_managers, load_managers_config

managers = build_managers(config_dict)            # from a dict
managers = load_managers_config("recipe.yaml")    # from YAML or JSON

managers.command.compute(state)        # publish commands first
obs = managers.observation.compute(state)
reward = managers.reward.compute(state)
result = managers.termination.compute(state)
```

`build_managers` rejects unknown manager keys and unknown term `func` names with
messages that list the valid options.

## Term library

Discover the registered terms at runtime with `list_terms()`. The first-class
locomotion library ships:

**Observation** (`base_lin_vel`, `base_ang_vel`, `projected_gravity`,
`joint_pos` (relative to default), `joint_vel`, `last_action`,
`velocity_commands`).

**Reward** (`track_lin_vel_xy_exp`, `track_ang_vel_z_exp`, `lin_vel_z_l2`,
`ang_vel_xy_l2`, `flat_orientation_l2`, `orientation_l2`, `dof_torques_l2`,
`dof_acc_l2`, `dof_vel_l2`, `action_rate_l2`, `joint_pos_limits`,
`joint_vel_limits`, `feet_air_time`, `feet_slide`, `alive`,
`termination_penalty`). Tracking terms return a bounded `(0, 1]` Gaussian
kernel; penalties return a squared magnitude (apply a negative weight).

**Termination** (`time_out` (truncation), `bad_orientation`,
`base_height_below_threshold`, `joint_pos_limit`).

**Command** (`uniform_velocity` - samples `[vx, vy, wz]` from uniform ranges and
resamples every `resampling_time` seconds; seed with `reset(rng=...)` for
reproducibility).

## Whole-body tracking (WBT)

Whole-body tracking trains a policy to imitate a reference motion. The terms
build on the same primitives as locomotion - they read the per-step target from
the `EnvState`, so the identical recipe runs on any backend.

A `MotionClip` is an immutable reference trajectory (joint positions, optional
velocities) sampled at a fixed `fps`, interpolated by *phase* in `[0, 1)`:

```python
from strands_robots.sim_managers import MotionClip
clip = MotionClip.from_arrays(frames_pos, frames_vel, fps=30.0)  # (num_frames, num_joints)
target_pos, target_vel, phase = clip.sample(t=0.5, loop=True)
```

The `motion_clip` **command** term replays a clip: each control step the
`CommandManager` advances its clock by `state.dt` and publishes the interpolated
target pose, target velocity, and phase into `state.extras` under the
`motion_target_pos` / `motion_target_vel` / `motion_phase` keys (see
`strands_robots.sim_managers.motion`). The WBT observation / reward / termination
terms read those keys, so a `motion_clip` command term must run **before** them
each step (the standard "commands first" ordering). A WBT term that runs without
one raises a clear error naming the missing command.

**Observation** (`motion_phase` - cyclic `[sin, cos]` of the phase;
`motion_target_joint_pos`; `motion_target_joint_vel`; `joint_pos_error` -
`joint_pos - target`).

**Reward** (`track_joint_pos_exp`, `track_joint_vel_exp` - bounded `(0, 1]`
Gaussian kernels of the per-joint pose / velocity error, same `track_*_exp`
convention as locomotion). For smoothness penalties reuse the shared
`action_rate_l2` / `dof_acc_l2` terms.

**Termination** (`motion_divergence` - failure when the L2 joint-position error
exceeds `threshold`, the standard "lost the motion" early stop).

```yaml
command_manager:
  terms:
    - {name: motion, func: motion_clip, params: {frames_pos: [[...]], fps: 30.0}}
observation_manager:
  terms:
    - {func: motion_phase}
    - {func: joint_pos_error}
reward_manager:
  terms:
    - {name: track_pos, func: track_joint_pos_exp, weight: 1.0, params: {std: 0.4}}
    - {name: track_vel, func: track_joint_vel_exp, weight: 0.2, params: {std: 2.0}}
termination_manager:
  terms:
    - {func: time_out}
    - {func: motion_divergence, params: {threshold: 1.5}}
```

Locomotion and WBT share the same managers, registry, and config DSL - they
differ only in which terms a recipe selects, so a multi-task policy reuses the
common terms with no code duplication.

## Worked example

`examples/sim_managers_locomotion.py` loads `sim_managers_locomotion.yaml`,
builds the four managers, and steps a Unitree G1 in headless MuJoCo - feeding
each step's physics state into the managers and printing the observation
dimension, per-term reward breakdown, and termination classification. Its
`_env_state_from_mujoco` helper shows exactly how a backend fills an `EnvState`
from a floating-base model/data:

```text
robot=unitree_g1  control_dt=0.0200s  observation_dim=99
episode ended at step 71: {'time_out': False, 'bad_orientation': True, ...}
per-term reward contribution (summed):
  dof_acc            -15.02715
  ang_vel_xy          -0.86686
  track_lin_vel       +0.30012
  alive               +0.21600
  ...
```

`examples/sim_managers_wbt.py` is the WBT counterpart: it builds a `motion_clip`
recipe and drives a position-controlled SO-101 arm in headless MuJoCo to imitate
the reference, recording the rollout to an MP4 while scoring it through the
tracking reward (the arm completes the clip with no divergence and a
pose-tracking-dominated reward).

## Extending

Register custom terms in any category with `@register_term(category, name)`;
they immediately become available to the config DSL. Because terms read only the
`EnvState` contract, a term written for MuJoCo runs unchanged on any other
backend that can populate the same fields.
