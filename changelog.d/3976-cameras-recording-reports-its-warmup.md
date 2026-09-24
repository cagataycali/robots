### Fixed: camera recording says whether it is capturing yet

`start_cameras_recording` waited for the recorder thread to warm its render
context but reported the outcome of that wait as a log-only warning, so a
caller read the same sentence over a capturing recorder and over one still
coming up, and a stop a moment later reported `0 frames - no clip written
(0 errors)` - which reads as "nothing went wrong". Start now names the warmup
it paid or says `NOT CAPTURING YET` and carries `capturing` / `warmup_s`,
`get_cameras_recording_status` marks the warming phase, and the zero-frame
stop line names the cause (warmup, render failures, or a synchronous recorder
nothing stepped).
