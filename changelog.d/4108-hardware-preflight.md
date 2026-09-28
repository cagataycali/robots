### Fixed: every hardware `Robot` rollout entry point runs one preflight

`start_task`, `execute_task`, `run_policy` and the agent tool's
`execute`/`start` pre-approval check now share `Robot._preflight` (shut down,
`duration`, `n_steps`, provider, port, required keywords, bus claim, in that
order). The tool's copy had drifted: it asked the operator to approve a
`lerobot_local` rollout with no checkpoint that the dispatcher then refused.
