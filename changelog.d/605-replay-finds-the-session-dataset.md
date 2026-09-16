### Fixed: `replay_episode` reads back what this session recorded

A dataset recorded with an `owner/name` id under `root=` and replayed by id
alone was looked up on the Hugging Face Hub and answered with the raw
multi-line 404. `replay_episode` now adopts the root this session's
`stop_recording` saved to (and says so); a Hub miss is one sentence naming the
local directory checked and the Hub. `get_recording_status` after a save
reports the dataset, frames, episodes and root instead of "last episode: 0
steps".
