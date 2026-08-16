### Fixed: a directly driven PolicyRunner refuses a recording rate it cannot describe

`start_recording(fps=...)` fixes the rate LeRobot timestamps every frame from
positionally, and the recorder is driven once per control step with no
decimation - so a rollout capturing at a different `control_frequency` can only
mislabel the episode. The engine's rollout entry points refuse that
disagreement, but `PolicyRunner.run` / `PolicyRunner.evaluate` are also driven
directly, with the entry-point guard off the path.

Driving the runner directly against an open 30 fps recording at
`control_frequency=50.0` wrote 20 frames declaring 0.0333s each for a capture
0.0200s apart - a 1.667x distortion - with the rollout and `stop_recording` both
reporting `status="success"`.

Both runner methods now apply the same check before any frame is written. They
raise `ValueError` rather than returning an error dict, matching
`_control_substeps`, whose raise is already documented as "the guarantee for
callers driving `PolicyRunner` directly" - a direct caller has no tool envelope
to read. The reason text is shared with the engine's error through a new
`dataset_rate_mismatch_reason`, so the two surfaces cannot describe the same
pair of rates differently. A rollout with no recording open, a matching rate,
and a backend that cannot record are all unaffected.
