### Fixed: the quickstart records a real arm with the tool that records

Step 1 of Getting Started asked the real robot tool to `start_recording`,
`teleoperate` and `stop_recording` - verbs it does not have (it publishes
execute, start, status, stop). The step now hands the agent
`lerobot_teleoperate`, which runs the recording session.

The real robot tool's unknown-action refusal names where the verb it was asked
for lives: the `lerobot_teleoperate` tool for teleoperation and recording, and
its own `execute`/`start`/`stop` for the simulation tool's
`run_policy`/`start_policy`/`stop_policy` - which it does have, under those
names. Any other spelling keeps the plain refusal, and building the refusal no
longer raises on an action value whose own rendering does.
