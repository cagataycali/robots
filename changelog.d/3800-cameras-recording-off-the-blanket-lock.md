### Fixed: `start_cameras_recording` / `stop_cameras_recording` work through the agent tool

Both verbs were dispatched under the action lock their recorder thread renders
under, so through `Robot(mode="sim")` start spent its whole readiness timeout
waiting on itself and stop's join expired with "did not stop within 5.0s" (or
reported an MP4 that was never written). They now run off the blanket lock, a
camera with no frames says "no clip written", and start says the recorder
samples wall time.
