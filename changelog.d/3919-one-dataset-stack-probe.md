### Changed: the sim backends share one dataset-stack probe

`start_recording` refuses the same way on MuJoCo, Newton and Isaac when
lerobot's dataset stack is not installed, and the message is unchanged - only
where it is built moves. The probe that resolves `DatasetRecorder` and turns a
missing extra into a named diagnosis was a verbatim copy in each backend, so the
diagnosis had to be changed three times; it now lives once on the recording
mixin all three inherit, with the plain-MP4 alternative each backend recommends
passed in.

One consequence for the layering: `DatasetRecorder` sits in `app`, and the
deferred read of it from `sim` is now a single edge from the shared mixin rather
than one per backend.
