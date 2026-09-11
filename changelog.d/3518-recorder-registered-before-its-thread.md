### Fixed: a camera recorder is registered before the thread that fills it

`Simulation.start_cameras_recording` started its capture thread and only then
published the recording, so for that window a thread was rendering into buffers
nothing could reach. Every other recorder verb reads that registration:
`get_cameras_recording_status` answered `[idle]` - the one reading it documents
it must never give - about a live capture, `stop_cameras_recording` returned
`status="success"` with "Was not recording cameras" and left the thread filling
its buffers to the `max_frames` cap, and the guard both start verbs ask found
nothing registered, so a second start was admitted onto the same cameras and had
its own registration overwritten a moment later, orphaning its thread and its
frames. The registration is now published first, which is the order the
synchronous recorder already used.

A capture thread that cannot be started is deregistered and reported in the
envelope, rather than raising past the tool boundary (its previous behaviour) or
leaving a registration that only a flush can clear.
