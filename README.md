# WBC auto-torque cleanup: measurement scripts

- `capture.py` runs two consecutive real WBC balance rollouts on ONE sim
  (Unitree G1, real SONIC `GR00T-WholeBodyControl-Balance.onnx`, MuJoCo headless
  EGL, 50 Hz x 150 ticks, `target_velocity = 0`) and dumps the pelvis-height
  trace plus a render after each. Run once per tree.
- `compose.py` builds the figure and asserts every number it renders: rollout 1
  byte-comparable across trees, rollout 2 differing on >10% of pixels, the
  excursion ratios, and the mutation counts.
- `mutate.py` is the mutation table: seven regressions of the hook's five no-op
  conditions and of the cleanup itself, each run against the 8 new cases and
  against the 490 pre-existing `tests/policies/wbc` cases.

Measured: rollout 1 identical on both trees (0.00% of pixels differ,
max|delta| = 1). Rollout 2 pelvis excursion 0.0339 m -> 0.1298 m on main,
0.0259 m here. 7 of 7 mutations caught by the new cases, 4 of 7 invisible to
the 490 pre-existing - including reverting the cleanup itself.
