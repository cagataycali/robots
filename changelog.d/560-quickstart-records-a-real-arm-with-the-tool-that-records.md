### Fixed: the quickstart records a real arm with the tool that records

Step 1 of Getting Started asked the real robot tool to `start_recording`,
`teleoperate` and `stop_recording` - verbs it does not have (it publishes
execute, start, status, stop). The step now hands the agent
`lerobot_teleoperate`, which runs the recording session. The real robot
tool's refusal of those verbs names `lerobot_teleoperate` (and the Python
`attach_teleop().teleoperate()` path) instead of only listing its own
actions.
