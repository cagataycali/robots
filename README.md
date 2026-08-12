# Isaac camera-readback pixel-floor artifact

Reproduces every number in the PR description.

* `census.py` / `matrix.py` - the coverage censuses that located the gap
  (`matrix.py` prints the shared-guard refusal matrix: which callers' refusal
  lines a test actually executed).
* `capture.py` - drives the four Isaac surfaces over the shared probe set,
  measures guard placement and the readback contract, and renders the MuJoCo
  sibling headless (`MUJOCO_GL=egl`).
* `mutate.py` - the eight-mutation table, run against this PR's module and
  against the pre-existing Isaac suite.
* `compose.py` - builds the figure and asserts every rendered value against
  `facts.json` before saving.
