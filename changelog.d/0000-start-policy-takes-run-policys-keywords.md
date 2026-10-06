### Fixed: `start_policy` takes every keyword `run_policy` takes

`start_policy` accepted 14 of `run_policy`'s 23 keywords, so a caller turning a
blocking rollout into a background one by renaming the call got a bare
`TypeError: MuJoCoSimEngine.start_policy() got an unexpected keyword argument`
for `control_substeps`, `n_episodes`, `reset_between`, `stop_when`, `observer`,
`async_rtc`, `rtc_inference_timeout_s`, `max_onframe_failures` or
`wbc_install_torque_control`. Both the base engine and the MuJoCo override now
take all of them and forward them to the rollout. On MuJoCo each one is checked
before the rollout is submitted, so a value `run_policy` refuses is refused by
`start_policy` with the same message instead of being lost on the worker
thread under a "Policy started" reply. The agent tool's descriptions of those
fields now name `start_policy` too.
