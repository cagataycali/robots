# run_policy rollout-knob pre-flight

Artifacts for the fix that checks `control_frequency` / `action_horizon` before
the `run_policy` tool's `overwrite=True` recording removes the caller's dataset.

* `capture.py`  - seeds a real one-episode LeRobot recording from a MuJoCo
  rollout, then re-opens the same `dataset_root` with `control_frequency=0.0`
  and records what survives. Run once per tree; each run prints the tree it
  resolved.
* `compose.py`  - builds the figure from the two dumps. Asserts every rendered
  number, that the two arms resolved different trees, and that the branch left
  the MP4 bit-for-bit unchanged.
* `probe_16_values.py` + `probe_main.json` / `probe_branch.json` - the full
  16-value sweep across both knobs (16 of 16 datasets destroyed on main,
  0 of 16 after).
