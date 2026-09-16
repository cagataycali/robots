### Fixed: `replay_episode` with `root` omitted finds the dataset this session recorded

Recording `lab/demo` to an explicit `root` and then replaying it without
`root` answered a raw huggingface_hub 404 for a dataset the sim had just
written. `start_recording` now remembers where each `repo_id` was recorded
this session, `replay_episode` replays from there when `root` is omitted
(saying so; an explicit `root` is never overridden), and a dataset that
cannot be opened gets a one-line cause plus the `root=` remedy.
