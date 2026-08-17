### Fixed

- **Local teleop now holds a leader frame to the same per-joint slew bound as the mesh receive path.**
  `teleoperate(publish=True)` drives a local follower and, from the same `get_action()` stream, every
  remote one, but only the mesh path bounded how fast a single joint could be commanded to travel
  (`STRANDS_MESH_INPUT_SLEW_ABS`) - so one device was judged by two rules and the follower next to the
  operator was the unguarded one. Measured on a MuJoCo Panda follower at 50 Hz, a leader sweeping at its
  servo maximum with one full-scale glitch frame had every frame applied, peaking at 140.9 units/s
  against a 25.13 bound. The merged frame is now checked against the same helper the mesh path calls, so
  the two cannot drift and the operator knob widens both. An over-speed frame is refused and counted in a
  new `slew_rejected` stat rather than clamped, since clamping would silently alter an actuator command;
  the bound sits above what a leader arm's own servos can produce, so a physical leader is unaffected.
  Refusals are counted apart from errors but still move the session off `success`, so a device whose
  units the bound does not expect cannot report a clean run while moving nothing.
