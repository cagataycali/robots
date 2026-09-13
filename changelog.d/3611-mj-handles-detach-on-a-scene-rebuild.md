### Fixed: `mj_model`/`mj_data` document that a scene rebuild detaches a cached handle

Both properties warned about one staleness mechanism -- racing a running
`PolicyRunner` worker's `mj_step`, described as values being "momentarily
stale", with "read between steps" as the remedy. A second mechanism was
undocumented and is worse: every op that recompiles the MJCF (`add_object`,
`add_camera`, `add_robot`, the `remove_*` family, `load_scene`) goes through
`spec.recompile`, which allocates a new model and data and installs them, so a
handle taken beforehand is not momentarily stale but detached for good. It keeps
its own sizes, state and clock, accepts writes and `mj_step` without error, and
neither its reads nor its writes reach the engine again; reading between steps
is no remedy, because the detached pair never agrees again.

The failure is silent in the shape that produces plausible numbers. Measured on
`so101` plus one `add_object`: the held model reports `nq` 6 against the live 13,
and a `qpos` written through the held data reads back 0.0 from
`get_observation`. On a ten-rung IK ladder the cached-handle run and the
re-read-handle run reported a byte-identical clearance sweep (-8.857 mm to
+32.371 mm) while the cached run rendered one unique frame out of ten and
`get_contacts` reported the cube untouched on every rung. Both properties now
name the triggering ops, the recompile that causes it, and the permanence, and a
regression test pins the runtime fact beside the documentation of it.
