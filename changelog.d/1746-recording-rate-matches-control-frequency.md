### Fixed: a dataset recording is refused at a rate its frames are not captured at

`start_recording(fps=...)` fixes the frame rate written into the LeRobotDataset
metadata, and LeRobot derives every timestamp from it positionally
(`timestamp = frame_index / fps`). The dataset recorder is driven once per
control step with no decimation, so the rate frames are really captured at is
the rollout's `control_frequency` - a differing `fps` could not be honored, only
mislabelled, and nothing compared the two.

The library defaults were exactly such a pair (`fps=30` against
`control_frequency=50.0`), so the documented record-then-rollout sequence
silently produced a distorted episode: frames captured 0.0200 s apart were
timestamped 0.0333 s apart, declaring a 1.30 s episode for a 0.78 s capture,
with `start_recording`, the rollout and `stop_recording` all reporting
`status="success"`. That per-frame interval is the control period a policy
trains on, and `replay_episode` derives its per-frame physics budget from the
dataset rate on the stated invariant that "the recorded control frequency IS the
dataset fps" - so the same episode also replayed at the wrong speed (measured on
a position-servo arm: record then replay round-tripped to 0.0000 rad at matching
rates and 0.0317 rad at the two defaults).

A rate disagreement is now refused before any frame is written, naming both
rates, the distortion factor and both remedies, from every rollout entry point
(`run_policy`, `eval_policy`, `evaluate_benchmark`, `start_policy`,
`run_multi_policy`). This matches the sibling rate guard in the same module,
which already refuses an `fps` that disagrees with the dataset on disk.
