### Fixed: a camera recording reported as stopped is one whose recorder thread has exited

`Simulation.stop_cameras_recording` asked the daemon recorder loop to exit,
joined it with a 5 s budget, and then discarded the outcome -- `Thread.join`
returns `None` whether or not the thread finished. A loop still inside `render`
when the budget expired therefore produced three answers that all read as
success: the MP4 was encoded from a frame buffer the live loop was still
appending to (the flush walks each buffer twice, so the encoded clip and the
reported frame count could describe different lists), the recording was
deregistered so no later call could re-join it, and `Already recording` no
longer refused a second recorder on the same cameras -- putting two capture
threads on one camera set. `get_cameras_recording_status` answered `[idle]`
about that live thread.

The join outcome is now read and reported. An expired join returns
`status="error"` with `stopped=False` and the per-camera buffered frame counts,
encodes nothing, and leaves the recording registered so a later
`stop_cameras_recording()` re-joins the loop and flushes it. Both
`start_cameras_recording` and `start_cameras_recording_synchronous` read the
recorder thread's liveness rather than the `running` flag the loop outlives, and
the status verb reports a `[stopping]` phase carrying `running` and
`thread_alive` separately.
