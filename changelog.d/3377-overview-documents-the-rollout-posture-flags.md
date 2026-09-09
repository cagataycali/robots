### Docs: `docs/simulation/overview.md` documents the rollout posture-flag domain beside the numeric ones

The page documents every numeric domain `SimEngine.run_policy` and
`SimEngine.eval_policy` hold their knobs to - `n_steps`, `duration`,
`action_horizon`, `n_episodes`, `max_steps` - naming each knob, its refusal and
where the check sits. #3368 bound the four posture flags in the same signatures
(`fast_mode`, `reset_between`, `wbc_install_torque_control`, `async_rtc`) to
the shared `boolean_flag_error` domain and added no paragraph, so a reader saw
the numeric knobs checked and had to infer the flags were not. The new
paragraph states what is checked, where (`run_policy`, `eval_policy`, MuJoCo's
`start_policy` before the submit, the `run_policy` tool before its recording
step) and where it is not (`PolicyRunner.run` does not repeat it). A test reads
the roster off both facade signatures and refuses a boolean parameter the
paragraph does not name, and pins the runner sentence to the runner's source so
a later decision to check there has to move the docs with the code.
