---
description: Stepping the simulation, reading contacts and forces, and commanding actuators.
---

# Physics and actions

Every parameter here obeys the three numeric domains in
[Simulation overview](overview.md), and every write is all-or-nothing.

## Physics

| Action | Key params |
|--------|-----------|
| `step` | `n_steps=1` (MuJoCo: max 100 000/call; Isaac and Newton have no ceiling). Non-negative whole number; `0` is an accepted no-op |
| `send_action` | `n_substeps=1` - **positive** whole number, no per-call ceiling (see [Actions](#actions)) |
| `set_gravity` / `set_timestep` | `gravity=[x,y,z]` or a scalar z-component; `timestep` |
| `get_contacts` / `get_contact_forces` | `get_contacts` lists every geom pair inside the detection range (`margin` + `gap`) and marks each `active`; only the pairs inside `margin` reach the solver, so a pair between the two thresholds is a proximity report carrying no force. Contact predicates count only `active` pairs; `get_contact_forces` gives the load a touching pair carries |
| `apply_force` | `body_name`, `force`, `torque`, `point` - latched on that body and re-applied every step until the next `apply_force` for it, so several bodies can hold wrenches at once (`force=[0,0,0]` stops one, `reset()` stops all) |
| `get_jacobian` | `body_name` *or* `site_name` *or* `geom_name`. Columns are DOFs of the whole compiled model, not the robot's joints: a free or ball joint owns several, and a two-robot scene reports one width spanning both. Pair `dq` with the `json` block's `dof_joint_names`, not with `robot_joint_names` |
| `get_mass_matrix` | - the reported `diagonal` is DOF-indexed on the same terms, named by `dof_joint_names` |
| `inverse_dynamics` | - compensation torques to hold the current `qpos`/`qvel` |
| `forward_kinematics` | `body_name` (optional) - refresh every body's pose from the current `qpos`, then filter to one body when named |
| `save_state` / `load_state` | `name` - snapshot/restore full physics. A checkpoint is valid only for the model it was taken against: any mutation that swaps the compiled model invalidates it, and `load_state` then errors instead of writing a state vector whose indices moved |
| `set_joint_positions` | `positions` (dict or ordered list), `robot_name`, `hold` - write `qpos` + run FK (teleport, bypassing actuators). Kinematic only: a joint held by a position servo is pulled back to that servo's setpoint by the next `step`, and the success text names those joints. `hold=True` moves the matching setpoints with the pose; a torque- or velocity-driven joint, and a joint a tendon couples to one `ctrl` (every stock gripper, the `stretch3` arm), is left alone because its `ctrl` is not a pose |
| `set_joint_velocities` | `velocities` (dict or ordered list), `robot_name` - write `qvel` directly |
| `get_energy` / `get_sensor_data` | `sensor_name` (optional) |

A body name for `forward_kinematics`, `get_body_state`, `get_jacobian`,
`apply_force` or `set_body_properties` may be bare (`gripper`) or namespaced
(`arm0/gripper`): `add_robot` namespaces every compiled body and the bare form is
retried under each robot's namespace. When that retry resolved the name, the
answer names the entity it landed on (`resolved 'base' to 'alice/base'`) and, in a
scene where several robots carry the name, the qualified spelling that reads each
other one. The physics *writes* refuse such a name outright rather than writing
the first robot attached.

Joint names come from `get_robot_state`, or from `robot_joint_names` - Python-only,
not in the tool schema's `action` enum. A dict key the model cannot resolve is
refused rather than skipped. Some assets name joints by servo id or CAD term (the
SO-101's are `1`..`6`, the SO-100's `Rotation`..`Jaw`); for those the registry
entry carries `joint_labels` - the `shoulder_pan` .. `gripper` the same arm's
driver and datasets use - so `get_robot_state` prints `1 (shoulder_pan)` and the
writers accept the label as a key, bare or `<robot>/<label>`, in any case.

A joint value is in that joint's own MuJoCo unit: radians for a hinge, metres for
a slide, never degrees. It matters most when mirroring a real arm onto its sim
twin, because the driver reports the other unit (`drivers/feetech` reads an SO-arm
in degrees), so a value outside the joint's range is refused naming the unit - and
names the conversion when converting *would* land inside it:
`shoulder_pan=-96.2 outside [-1.92, 1.92] rad (radians, not degrees: -96.2 deg =
-1.679 rad)`.

## Actions

`send_action(action, robot_name=None, n_substeps=1)` writes actuator/joint targets
and advances physics. `action` accepts either form:

| Form | Binding |
|------|---------|
| `{joint_or_actuator_name: value}` mapping | applied by name; unresolved keys are reported in an `unresolved_keys` JSON block so a caller can self-correct (no silent drop). Use a mapping to target a subset of actuators |
| ordered numeric vector (`list` / `tuple` / 1-D `numpy` array) | bound positionally to `robot_action_keys(robot_name)` in declaration order - the convention `replay_episode` uses too - so a policy's raw chunk drives the arm without being zipped into a dict. The length must match the actuator count exactly; a mismatch, or a non-numeric / scalar / string `action`, returns a structured error naming the count and order |

The vector binds to `robot_action_keys`, not `robot_joint_names`: those are the
keys `send_action` resolves and the order the `LeRobotDataset` recorder writes the
`action` column in. The two coincide unless a robot has passive/mimic joints, a
tendon gripper, or a floating base on the Newton backend - whose 6-DoF free joint
is a joint but not a commandable scalar, so it is absent from the action keys (its
pose is read as `base_pos` / `base_quat` / `base_lin_vel` / `base_ang_vel`) and
`send_action` refuses it as a command key. Do not assume equal widths.

`n_substeps` is how many physics steps the written targets are held for: a
**positive** whole number, coerced from a NumPy or integral-float count
(`np.int64(3)`, a `3.0` from a config), refused with nothing written when
fractional, zero, negative, non-finite, boolean or non-numeric. The floor is `1`
rather than `step`'s `0` because of that write - to advance without commanding,
use `step(n)`.

## See also

- [Simulation overview](overview.md) - the scene-construction and rendering verbs.
- [Policy rollouts](rollouts.md) - driving these actions from a policy.
