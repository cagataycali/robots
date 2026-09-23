### Fixed

- The simulation's out-of-range action warning now reads the range that actually
  holds the command. MuJoCo enforces a command two ways: it clamps a
  `ctrllimited` actuator's `ctrl` inside `mj_step`, and for an unlimited position
  servo it clamps nothing at all - but that actuator's `ctrl` IS the joint
  target, so the joint's own range bounds the pose it reaches. The warning read
  only the first, which skipped every so101 actuator, since the shipped MJCF
  authors neither `ctrlrange` nor `inheritrange`: a degrees-valued action chunk
  applied as radians left five of six joints pinned at a limit while
  `send_action` reported success and nothing was logged. The effective-range rule
  now lives once, in `strands_robots.simulation.mujoco.scene_ops.effective_ctrl_range`,
  where `set_gripper` already resolved the same bounds, and the warning names
  which source it read and which mechanism held the value.
