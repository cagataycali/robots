### Fixed: `get_recording_status` reports the dataset's episodes, not just the open buffer

The status text read the open-episode mirror alone, which empties at every
flush, so right after `run_policy(n_episodes=3)` had saved 45 frames it said
"0 steps captured" - while `stop_recording` on the next call reported "45
frames, 3 episode(s)". The saved episode and frame counts now stand beside the
open buffer, in the text and in the json (`episodes_saved`, `frames_saved`,
`open_episode_index`). The two recipes that still promised `run_policy` "once
per episode" name the published boundaries instead. Behaviour is unchanged.
