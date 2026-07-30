### Fixed: send_action refuses an action value that is not finite

`SimEngine._coerce_action` validated that every action value coerces to a scalar
`float` and that an ordered vector's length matches the robot's actuator count,
but not that the value is finite -- and `nan`/`inf` are perfectly good `float`
objects. A non-finite value was therefore admitted, written to `data.ctrl` and
handed to the integrator, with `send_action` reporting `status="success"`. The
ctrl-clamp warning passed it through as well, and attributed a `nan` to clamping.

`nan` is not clamped. MuJoCo finds the resulting non-finite `qacc` and resets the
world to its initial pose -- on every substep, so the whole call's physics is
discarded, and for *every* robot in the scene rather than only the commanded one.
Measured on two Panda arms parked at `joint1=0.9000`, one `nan` sent to the first
arm alone left both reporting `joint1=0.0023`; the arm that was never addressed
had been teleported home. Because any later finite command integrates normally
the teleport leaves no residue, so a recording rollout reports success for every
step and the dataset simply holds a trajectory no robot followed. `inf` *is*
clamped into `ctrlrange`, i.e. silently rewritten into a full-travel command:
the same `joint1` was driven to its limit of `2.8973`.

Both accepted action shapes are now held to the shared rule the sibling state
writers `set_joint_positions` / `set_joint_velocities` already applied, so a
mapping value and a vector entry cannot diverge on it. A mapping error names the
offending key; a vector error names the position and the actuator key it binds
to. The refusal happens before any actuator is written, so a good key alongside a
non-finite one applies neither. The accepted domain is finiteness alone: a
numeric string remains an accepted spelling of a scalar, and a finite magnitude
outside `ctrlrange` remains a units question surfaced by the clamp warning.
