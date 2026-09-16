### Fixed: `get_recording_status` reports the dataset a save wrote, not the buffer it cleared

`stop_recording` answered `lab/ep -- 5 frames, 2 episode(s)` and the very next
`get_recording_status` answered `[idle] Not recording (last episode: 0 steps)`.
`save_episode` mid-session was the same one function earlier: `Episode 1 saved
-- 3 frames`, then `[recording] 0 steps captured`. Both writers reset the
`trajectory` mirror, and that buffer was the only thing this reader counted, so
every SUCCESSFUL save reported nothing recorded - at the one moment a caller
polls to confirm the save.

Both writers now stash what the dataset holds (id, root, frames, episodes), and
the status reports it plus the `replay_episode(repo_id=...)` call that reads it
back - verified by parsing that call out of the message and running it against a
real MuJoCo recording. A session that has saved nothing says so instead of
reporting a zero-step episode, and the reply gains a json block. The four facts
are stashed together rather than read from the `last_dataset_root` pair
`start_recording` writes, which moves at the next `start_recording` and would
name a dataset the counts were never written to.
