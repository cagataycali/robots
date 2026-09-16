### Fixed: recording texts tell the agent where `run_policy` frames go

`start_recording` promised "run_policy (one call per episode)", but a single
`run_policy` buffers into the open episode and flushes nothing, so two
demonstrations became one dataset episode; `get_recording_status` read the
open-episode buffer alone and said "0 steps captured" right after three
episodes were saved. The `run_policy` answer now names the open episode and
the ways to close it (`reset`, `run_policy(n_episodes=N)`, `stop_recording`),
the status reports saved episodes and frames beside the open buffer (with a
json block), and the recipes are corrected. Behaviour is unchanged.
