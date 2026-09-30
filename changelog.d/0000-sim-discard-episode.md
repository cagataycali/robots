### Added: `discard_episode` drops a bad take before it reaches the dataset

A sim recording session could close an episode only by saving it (`reset`,
`save_episode`, `stop_recording`), so a take that went wrong had to be written
and then deleted with LeRobot's dataset tools. `discard_episode()` - a Python
method on every backend's recording mixin and a published MuJoCo tool action
(78 actions now) - clears the open episode's buffered frames and un-counts them,
leaving saved episodes untouched; call `reset` next for the retake. It is
refused outside a recording and while a policy is writing the buffer, and a
LeRobot without a buffer-clear surface is reported rather than pretended.
