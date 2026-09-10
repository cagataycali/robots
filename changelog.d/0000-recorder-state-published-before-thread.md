### Fixed
- `Simulation.start_cameras_recording` publishes `_cams_rec_state` before starting the recorder thread, so a reader that reaches the recorder through the attribute (a render hook, a stop racing the start) sees the same state the thread is using instead of `None`.
