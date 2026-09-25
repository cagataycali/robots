---
description: Running, stopping, evaluating and watching a policy in simulation.
---

# Policy rollouts

| Action | Key params |
|--------|-----------|
| `run_policy` | `robot_name` (required), `policy_provider="mock"`, `policy_config={}`, `policy_object=None`, `instruction=""`, `duration=10.0`, `control_frequency=None`, `action_horizon=8`, `n_steps=None`, `seed=None`, `async_rtc=None`, `rtc_inference_timeout_s=None`, `stop_when=None`, `observer=None` ([observers](observers.md)), `video=None` |
| `start_policy` | same args, async/non-blocking |
| `stop_policy` | `robot_name` (optional, defaults to `""` - every rollout) |
| `list_policies_running` | - |
| `run_multi_policy` | `policies={robot: Policy}`, `instructions`, `duration`, `n_steps` |
| `eval_policy` | `robot_name` (optional; auto-resolves the sole robot), `n_episodes=1`, `max_steps=300`, `success_fn=None`, `async_rtc=False`, `rtc_inference_timeout_s=None`, `video=None` |
| `evaluate_benchmark` | `spec` or a registered name, `n_episodes`, `video=None` |
| `replay_episode` | `repo_id`, `robot_name=None`, `episode=0` |

`run_policy` / `eval_policy` / `run_multi_policy` bind the policy's output keys to
the robot's *action* keys via `set_robot_state_keys(robot_action_keys(robot_name))`:
keying by `robot_joint_names` would emit keys that resolve to nothing and leave
those DOFs unmoved (see [Actions](physics.md#actions)). Read the keys a robot
expects with `get_features(robot_name=...)`. A `stop_when` clause, and a benchmark
spec's clauses, are [predicates](predicates.md). Every knob below is documented in
full in the `run_policy` / `eval_policy` docstrings, and what a rollout hands
back is [Rollout results](rollout-results.md).

## Refusals

Each parameter is checked at the entry point - `start_policy` synchronously, before
the background rollout starts, so a malformed request never returns a false
"started" - together with the provider's own class-level `preflight` hook, and ahead
of the robot claim, so a refused call leaves the robot startable.
`PolicyRunner.run` / `PolicyRunner.evaluate` are drivable directly and raise
`ValueError` there instead, having no envelope to report through.

| Parameter | Domain |
|-----------|--------|
| `n_steps`, `duration`, `control_frequency` | positive. The horizon is `duration` (seconds) or `n_steps` (`duration = n_steps / control_frequency`); `n_steps` wins when both are set, `max_steps` is a legacy alias |
| `action_horizon` | positive whole number - actions consumed from each chunk before it is re-queried. `run_multi_policy` also takes per-robot mappings (`{robot: horizon}`, `instructions={robot: text}`) whose keys must name a robot of that call |
| `n_episodes`, `max_steps` | positive whole number; `max_steps` only when it is the horizon actually read, since a `spec=` call takes its horizon off the benchmark |
| `rtc_inference_timeout_s` | positive finite seconds, or `None` to wait without a deadline |
| `policy_config`, `policy_kwargs` | `dict` - splatted into `create_policy` and into every `get_actions` call |
| `policy_object` | a `Policy` instance, so a provider name or the class is refused by name rather than as an `AttributeError` one layer down |
| `observer`, `on_frame`, `success_fn` | callable, refused before the first step |

The four **posture** flags select a branch rather than scale a quantity -
`fast_mode` (pace the loop at `control_frequency` or run it unpaced),
`reset_between` (reset the scene between episodes or carry the end state over),
`wbc_install_torque_control` (install the WBC torque shim for the call or leave the
actuators alone) and `async_rtc` (overlap inference with actuation or drain each
chunk first) - so a non-boolean is refused rather than read by truthiness:
every non-empty string is truthy, so `fast_mode="false"` would run unpaced, and
`async_rtc="false"` would report `rtc_async_enabled=True`, under `status="success"`. The domain is
the shared `boolean_flag_error` one the recording postures and the mesh wire schema
use, bound to the tool-error envelope through `SimEngine._validate_posture_flags`
and checked ahead of robot resolution, so a refused call builds no policy and
touches no scene. `run_policy` checks all four and treats `async_rtc=None` as its
"resolve from the policy" spelling; `eval_policy` declares `async_rtc` as a plain
`bool` and refuses `None`. `wbc_install_torque_control` is declared by all three
rollout surfaces - `run_policy`, `eval_policy` and `evaluate_benchmark` - which
install the controller through the one reader
`SimEngine._install_action_controller`, so a scored rollout drives the scene the
way an unscored one does. Unlike the numeric knobs the check sits at the facades
only: `PolicyRunner.run` takes these flags as the facades hand them and
does not repeat it.

## Stopping, and what counts as running

`stop_policy` is honoured at any point after `start_policy` returns - before the
rollout's first frame, and while it is still queued behind a busy executor. Its
verdict comes from the same in-flight population `list_policies_running` reads, so
the two never report opposite facts about one robot at one instant, and that
population counts either launch shape: a blocking `run_policy` registers no future,
yet is reported as running, is named, and is halted by a stop carrying no
`robot_name` (the shape the mesh e-stop fanout broadcasts). `Was not running on
'<robot>'` is reserved for the genuinely idle case; a stop that finds its own robot
idle while another is mid-rollout names that rollout and the call that ends it, and
carries the reason when that robot's last rollout ended in error.

`list_policies_running` answers on every backend from that population - MuJoCo,
Newton, Isaac and a peer polled over the mesh all name the robots they are driving
at that instant. A backend that can report no population at all is refused rather
than reported as idle: "no policies running" is an affirmative claim about every
robot in the world.

Scene mutations read the same population: `add_robot`, `remove_robot`, `add_object`,
`remove_object`, `move_object`, `add_camera`, `remove_camera`, `load_scene`,
`set_gravity`, `set_timestep` and `reset` refuse while a rollout is driving, naming
the robots in flight and the `stop_policy` remedy, because swapping the compiled
model under a live rollout segfaults. The rollout's own driving thread is exempt, so
a multi-episode rollout still resets between its own episodes.

## See also

- [Simulation overview](overview.md) - the scene-construction and rendering verbs.
- [Rollout results](rollout-results.md) - the result payload and the async-RTC telemetry.
- [Physics and actions](physics.md) - what a rollout writes every control step.
- [Predicates](predicates.md) - `stop_when` and benchmark clauses.
- [Rollout observers](observers.md) - watching a rollout step by step.
- [Policy providers](../policies/overview.md) - what `policy_provider` may name.
