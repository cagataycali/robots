### Fixed: camera recording says when the recorder is still warming up

`start_cameras_recording` returned the same success sentence whether its
thread was capturing or still bringing up its GL context (the not-ready
case was a log-only warning), and a stop a few seconds later listed
`0 frames … -> rec__cam.mp4` for a file that was never written. Start now
reports `capturing` and either the warmup time or NOT CAPTURING YET, status
marks the warming phase, and stop says "no MP4 written" and why (artifact
`path` is null).
