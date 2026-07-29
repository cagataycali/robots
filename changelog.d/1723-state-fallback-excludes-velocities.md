### Fixed: the observation-derived state ordering no longer interleaves velocity siblings

`LerobotLocalPolicy._resolve_state_order` falls back to the observation's own
scalar keys when the configured `robot_state_keys` match nothing in it -- the
generic `joint_0..joint_N` vs named-joint mismatch. It took those keys in the
observation's insertion order, and the MuJoCo backend emits a velocity sibling
beside every joint position (`simulation/mujoco/rendering.py`: `obs[jnt_name]`
then `obs[f"{jnt_name}.vel"]`), so that order alternates
`[pos0, vel0, pos1, vel1, ...]`.

The result was twice the DOF count, then truncated to the model's declared
`observation.state` dim -- so HALF the slots held velocities and the trailing
joints were dropped entirely. On a 6-DoF arm driven to distinct known values the
resolved order was `['shoulder_pan', 'shoulder_pan.vel', 'shoulder_lift', ...]`
and the state vector fed to the model was
`[0.2981, -0.4700, -0.1973, 0.6561, 0.4978, -0.5439]` against correct positions
`[0.2981, -0.1973, 0.4978, 0.0987, -0.3974, 0.2474]`, with `wrist_flex`,
`wrist_roll` and `gripper` absent. Nothing raised and nothing logged about it:
the policy ran on a wrong state vector while reporting success. The producer
documents `<name>.vel` as an ADDITIVE key that position-only consumers are
unaffected by, and this consumer was the exception.

An ordering derived from the observation now drops each `<joint>.vel` whose
`<joint>` position companion is present. A `.vel` key with NO companion is kept,
because some embodiments legitimately declare velocity state -- `embodiments.json`
gives LeKiwi body-frame base velocities `x.vel` / `y.vel` / `theta.vel` with no
`x` / `y` / `theta` position key -- so pairing is decided per key rather than by
suffix. An explicitly configured `robot_state_keys` ordering is returned
untouched: an operator naming `elbow.vel` is stating the model's input, and this
only cleans up an ordering inferred from whatever the observation happened to
contain. The mismatch warning and its `generic_state_keys_used` telemetry are
unchanged, so the misconfiguration that reaches the fallback is still surfaced.
