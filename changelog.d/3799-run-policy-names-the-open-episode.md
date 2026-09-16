### Fixed: `run_policy` while recording reports the open episode

A single-rollout `run_policy` during a dataset recording now says how many
frames sit in the open (unsaved) episode, says APPENDED when an earlier rollout
already left frames there, and names the boundaries an agent can reach
(`reset` between rollouts or `n_episodes=N` in one call). Two back-to-back
rollouts silently merged into one episode; the advice pointed at
`save_episode`, which is not in the published action enum.
