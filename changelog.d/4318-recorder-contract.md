### Fixed: a closed dataset recorder refuses a frame instead of swallowing it

`DatasetRecorder.add_frame` returned without a word once the recorder was closed
- after `finalize()`, or after a `save_episode()` that failed and closed it - so
every later frame was neither written, nor raised as `RecordingFrameError`, nor
counted in `dropped_frame_count`. A closed recorder now treats that frame as a
lost one: it raises `RecordingFrameError` (the default, `strict=True`) or counts
the drop and logs it (`strict=False`).

The recording contract is now written down as the `strands_robots.recorder.Recorder`
ABC - `add_frame`, `save_episode`, `clear_episode_buffer`, `finalize` and the
read-only `closed` property. `DatasetRecorder` implements it; nothing else about
its API changes.
