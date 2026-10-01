### Added: an embodiment can speak DROID - joint-velocity actions and a gripper fraction (`panda_droid`)

π0.5-DROID (`lerobot/pi05_droid`) conditions on 7 arm joints + the gripper as the
fraction closed and emits 7 arm joint velocities + that fraction; strands had no
velocity action mode, no gripper conversion outside the SO arms' `RANGE_0_100`, and
refused a padded 32-wide action against 8 action keys, so it took an out-of-tree
adapter to run. `EmbodimentMap` gains `action_mode` (`"position"` | `"velocity"`,
integrated from the measured joints at `action_dt` seconds per action),
`gripper_fraction` (`[open, closed]` of the gripper column's joint),
`gripper_followers` (more action keys driven by that one gripper dimension) and
`action_dim_policy` (`"strict"` | `"truncate"` for a padded model width), each graded
at construction. The shipped `panda_droid` embodiment (alias `franka_droid`) uses
them: `create_policy("lerobot_local", pretrained_name_or_path="lerobot/pi05_droid",
embodiment="panda_droid")` drives the Isaac Panda for 2x150 steps and records 300
frames with no adapter.
