### Fixed: an Isaac camera recording that cannot be encoded is kept, not discarded

`IsaacSimulation.stop_cameras_recording` deregistered the recording before it
read a single buffer, so the one failure it reports as an error - no MP4 encoder
installed, which `encode_clip` raises before opening any writer, having encoded
nothing and touched nothing - destroyed every camera's frames. The refusal quoted
the encoder's own remedy ("install it, call again"), and there was nothing left
to call it on: a second `stop_cameras_recording()` answered `"Was not recording
cameras."`, and a new `start_cameras_recording` took the slot under
`status="success"`, silently dropping the buffers.

Now only a flush that encoded deregisters - the rule the MuJoCo recorder already
applied through `_flush_and_deregister_cameras_recording`, and which
`docs/recording.md` documented for both. A flush with no encoder returns
`stopped: False` plus the per-camera buffered counts and leaves the recording
registered, so installing the encoder and calling the verb again writes the MP4s;
a start is refused for as long as those frames are unencoded. Capture stops
either way, so a retained recording cannot grow.

Both recorders now word that one absence from one place,
`strands_robots.simulation.recording.encoder_absent_flush_refusal`, which is what
had let the rule hold for one of them and not the other.
