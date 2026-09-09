### Fixed: the SO-101 reference pick reports a refused primitive instead of a 0 mm lift

`examples/18_so101_pick_and_lift.py` composed ten simulation primitives and
discarded every envelope, returning a hard-coded `status="success"`. With the
`[sim-mujoco]` IK solver absent all three `move_to` calls are refused - each
naming the install that fixes them - and the run still summarised a completed
pick and printed `PICK FAILED - cube lifted only 0.0 mm`. That is the one
outcome the module docstring spends three paragraphs saying to expect (a
friction pinch holds nothing, "0 mm lift with the fingers in contact"), so a
reader who hits the dependency gap concludes the reference is the known-broken
friction case and never sees the remedy the library already produced.

Each call now goes through `_ok()`, which raises on an error envelope;
`run_pick()` catches it once and returns `status="error"` carrying the refused
step and the refusal's own text, and `main()` prints it and exits non-zero. The
successful path is unchanged - the same rollout still lifts the cube 150.3 mm.
