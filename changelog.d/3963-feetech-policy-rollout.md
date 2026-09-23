### Added: the Feetech SO arms run a policy

`FeetechDriver.run_policy` / `start_task` roll a policy out on an SO-100, SO-101
or LeKiwi at 30 Hz (`control_frequency=` to change it) instead of refusing, so a
native-driver SO arm is no longer read-only to a policy. Each step reads the whole
arm in one sync-read, hands the policy lerobot's own `{"<joint>.pos": degrees}`
observation and commands its action through `send_action`; a setpoint the bus
refuses ends the rollout carrying that refusal as its `exit_reason`, and
`get_task_status` reports the snapshot after the thread is gone. `stop_task`
halts the loop and leaves the arm energized (`stop` still de-energizes), and
reports `stopped=False` rather than claiming a halt when a blocking policy keeps
the thread in the loop.

The loop itself is one class, `strands_robots.drivers.rollout.PolicyRollout`,
moved out of the UR driver and parameterised by how a robot is read and
commanded - so the two drivers cannot drift apart on what `exit_reason` means.
The provider build both `start_task`s perform is shared with it as
`policy_from_provider`.
