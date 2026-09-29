### Changed: MuJoCo `step()` under an open recording writes through `RecordedFrame` too

`step()` records the whole scene at the dataset rate while a recording is open,
and it was the last recording path still assembling the dataset frame itself
(the `<robot>__<key>` prefixing, the scoped camera arrays and the required
action columns). It now hands its state, the actuator targets in force and its
camera arrays to `strands_robots.simulation.recording.RecordedFrame`, the writer
every rollout hook and `run_multi_policy` loop already use. The recorded frames
are unchanged.
