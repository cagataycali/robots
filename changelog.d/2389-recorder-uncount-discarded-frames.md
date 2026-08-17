### Fixed

`DatasetRecorder.clear_episode_buffer` no longer leaves the frames it discarded
counted in the cumulative `frame_count`. `add_frame` counts a frame when it
buffers it, so an aborted episode - the path `run_multi_policy` takes when a
rollout bails mid-episode - used to inflate every later
`save_episode()['total_frames']` and `push_to_hub()['frames']`, and blinded
`stop_recording`'s "captured no frames" refusal into reporting success for a
dataset holding only `meta/info.json`. When the buffer could not be discarded
the frames are still queued for the next `save_episode`, so both counters are
left describing them.
