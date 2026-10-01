### Fixed: the `run_policy` tool judges and builds the policy before it overwrites the dataset at `dataset_root`

`tools.run_policy` with `dataset_root=` started its recording with
`overwrite=True` and only then let each episode's `Simulation.run_policy`
resolve the provider, run its preflight and build the policy, so an unknown
provider name or a `lerobot_local` checkpoint id that does not exist was
discovered after the dataset already at that root had been replaced with an
empty one (two recorded episodes became `total_episodes=0` under `0/1 episodes
ok`). When a recording is requested the tool now resolves the provider, runs
the shared `preflight_reason` against the simulation's observation keys and
calls `create_policy` once, all before `start_recording`; each refusal is the
tool's `status=error` envelope naming the reason and the untouched root, and
the one built policy is forwarded to every episode as `policy_object=` (a
checkpoint load is paid once per call, not once per episode). The
recording-less path forwards exactly what it did before. The summary line now
quotes the first failed episode's reason. (#4155)
