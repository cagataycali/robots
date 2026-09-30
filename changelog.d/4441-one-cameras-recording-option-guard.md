### Changed: one camera-recording option guard serves MuJoCo and Isaac

`start_cameras_recording` on MuJoCo (both the daemon-thread and the
synchronous recorder) and on Isaac now run the same pre-flight,
`strands_robots.rendering.video.cameras_recording_option_error`, instead of one
private copy per backend. The accepted values and the refusal text are
unchanged; a future fix to what counts as a usable `fps`, frame cap or pixel
count now lands on every backend at once.
