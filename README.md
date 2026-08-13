# MuJoCo motion-primitive resolution refusals

Artifact for the PR pinning every "cannot resolve what to drive" refusal in
`move_to` / `set_gripper` / `rotate_wrist`.

* `capture.py` - renders the three scenes headless (`MUJOCO_GL=egl`) and writes
  `facts.json`. Framing gates are assertions inside the script: the honored
  `move_to` must change >10% of pixels, and each refusal must change exactly 0.
* `compose.py` - builds the figure; every drawn number is asserted against
  `facts.json` and the layout pitches are derived, not guessed.
* `mutate_resolution.py` - the 7-mutation table, run against this module and
  against the 163 pre-existing primitive cases. Anchors are AST-scoped to the
  enclosing function (two of them appear twice in the file) and the source is
  restored byte-identically.
* `facts.json` - the measurements.
