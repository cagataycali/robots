### Fixed: `target_velocity` is bounded by one locomotion envelope on the mesh and in the WBC policy

A mesh `execute` / `start` payload's `target_velocity` was coerced into the
`+/-1e6` domain it shares with `target_pose` - a coordinate range, not a speed -
and `WBCPolicy._validate_velocity` checked finiteness only before the value was
multiplied by `cmd_scale` into the observation, so `[1e6, 0, 0]` was a valid
command at both layers while sibling controls in the same validator were
tightly bounded (F-005, CWE-20). Both now hold every component to
`strands_robots.locomotion_envelope` (one stdlib-only definition: ±2.0 m/s
linear, ±2.0 rad/s angular, operator-raisable through
`STRANDS_MAX_TARGET_LINEAR_VELOCITY_MPS` / `STRANDS_MAX_TARGET_ANGULAR_VELOCITY_RPS`).
A component past the bound is refused with a reason naming the component,
value, bound, unit and knob - never clamped.
