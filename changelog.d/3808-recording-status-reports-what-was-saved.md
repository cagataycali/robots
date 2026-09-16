### Fixed: `get_recording_status` reports the dataset that was saved, and every reply carries a json block

Measured on `Robot("so101", mode="sim")`: `start_recording(repo_id="lab/so101_ep",
root=/tmp/x)` then 30 scripted steps then `stop_recording` saved 37 frames in 1
episode, and `get_recording_status` answered `[idle] Not recording (last episode:
0 steps)` - byte-identical to the sentence a sim that has never recorded gives.
It read the trajectory mirror that `stop_recording` clears when it releases the
recorder, so the count was structurally zero after every successful save. Two
readers were sent here for exactly that number: `docs/simulation/overview.md`
advertises "Episode, frame count, output dir" for this method, and
`docs/troubleshooting.md` sends an operator here to check the frame count when an
MP4 is empty - so the zero falsely confirmed the "stopped before any frames"
diagnosis it was meant to rule out.

`stop_recording` now stashes the id, root, frame count and episode count of the
save it just made, past that teardown, as one mapping - a reader must never pair
one save's id with another's counts. Idle reports them (`Last saved: lab/so101_ep
- 37 frames, 1 episode(s) at /tmp/x (replay_episode(repo_id='lab/so101_ep',
root='/tmp/x') reads it back)`) and says `nothing saved in this session` before
any save. An open session names the dataset it is writing into. The printed
replay call carries `root=` because a bare id is not always enough: a later
session that saves nothing still moves the id/root stash that replay's root
adoption reads, and a bare-id replay of the older dataset then errors while the
printed call replays all 37 frames. Every branch - no world, idle, recording -
now also returns a json block of one shape (`world`, `recording`, `steps`,
`last_save`, plus `repo_id`/`root` while recording), so a poller reads a mapping
instead of parsing three sentences.
