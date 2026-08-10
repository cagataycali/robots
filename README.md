# robot_name auto-resolution artifact

Generated headless on MuJoCo (`MUJOCO_GL=egl`) for the PR pinning the motion
primitives' documented `robot_name` default.

* `capture.py` - renders the three scenes and records every measured fact to JSON.
* `compose.py` - composes the figure; asserts every number it prints.
* `mutate_new_tests.py` / `mutate_existing_tests.py` - the mutation table: four
  mutations of the shared `_primitive_resolve_robot` preamble, run against the
  new tests and against the four existing primitive test files.
