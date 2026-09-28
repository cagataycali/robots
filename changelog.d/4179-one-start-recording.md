### Changed: `start_recording` is written once, on `DatasetRecordingMixin`

The MuJoCo, Isaac and Newton backends each carried their own `start_recording`
(493, 336 and 348 lines), so every refusal it makes - the rate, the posture
flags, the camera list, colliding camera columns, a live session, the wipe
ordering - was written and graded three times. The method now lives once on
`strands_robots.simulation.recording.DatasetRecordingMixin`; a backend supplies
only the schema its scene declares (`_collect_recording_schema`, returning a
`RecordingSchema`) and its camera names. The call, its arguments, its refusals
and its replies are unchanged, with one parity fix: resuming a Newton dataset
now compares the action columns against the scene, as MuJoCo and Isaac already
did.
