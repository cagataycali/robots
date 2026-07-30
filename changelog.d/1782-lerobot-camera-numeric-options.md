### Fixed: the camera tool refuses a numeric option it cannot honor

`lerobot_camera` validated its `save_path` at the boundary and forwarded every
numeric option raw, into a camera configuration, an MP4 container header and a
capture loop's bound. `capture_duration <= 0` made the recording loop's bound
`int(fps * capture_duration)` zero, so the loop body never ran - and the tool
returned `status="success"` with a summary whose `Saved:` line named a 258-byte
MP4 that no decoder will open, a recording reported as complete that contains no
video. `capture_duration=nan` leaked `cannot convert float NaN to integer`, an
`int()` internal naming neither the tool nor the parameter, and left that stub
behind; `capture_duration=True` silently recorded one second. `fps` in
`{0, -10, 2.7, nan, inf, True}` was refused only by the camera driver, which
compares the requested rate against the rate the attached device reports - so a
value impossible on every camera was reported as a property of this one, after
the device had already been opened and reconfigured.

Each option is now checked against the shared domain for its kind before a
camera is opened: `width` / `height` / `fps` count pixels and frames
(`positive_whole_number_error`, which already owns the recorders' geometry and
rate), and `capture_duration` / `preview_duration` / `timeout_ms` are continuous
spans of time (`positive_finite_number_error`). Only the options an action
actually consumes are checked, so `record` still accepts a `timeout_ms` it never
reads (its asynchronous read passes a fixed one), `timeout_ms` is effective only
under `async_mode`, and `discover` / `list` are unaffected. Because the geometry
shares the recorders' domain, an integral float is now honored rather than
refused: `width=640.0` reaches the camera and the container as `640` instead of
dying in an OpenCV overload-resolution dump.
