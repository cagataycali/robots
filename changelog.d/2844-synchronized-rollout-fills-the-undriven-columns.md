### Fixed: a synchronized rollout that drives a subset of the scene records the rest as measurements

`start_recording` declares `observation.state` over EVERY robot in the scene,
prefixing each column with its robot's name (`alice__shoulder_pan`). A rollout's
recording hook supplies only the observation of the robots it drives, so every
column belonging to another robot is absent from the frame and
`DatasetRecorder.add_frame` writes it as `0.0` - in the same column, with the
same dtype, as a measurement, for every frame of the episode, under
`status="success"`. Nothing downstream can tell that zero from a real reading: a
policy trained on the dataset learns the other robot is permanently at its zero
pose. `undriven_robot_state` exists to fill those columns from the engine
instead, and the three single-policy recording hooks went through it.

`run_multi_policy` did not. Its merge loop iterates the keys of `policies`, and
`SimEngine.run_multi_policy` requires each key to name a robot in the scene
without requiring the keys to cover it - the constraint runs one way only. So a
synchronized call that drives a subset of the scene left every other robot to
the fill, exactly as a single-policy hook did before the helper existed. The
helper's own docstring credited the synchronized loop with already having the
guarantee, on the strength of its merging every robot *it drives*.

Measured on a two-robot MuJoCo scene with `bob` parked at `shoulder_pan 0.9`,
`elbow -0.7` and `policies={"alice": MockPolicy()}`: the schema declared
`bob__shoulder_pan` and `bob__elbow`, the rollout reported `status="success"`
with 6 frames and a written MP4, and both of `bob`'s columns were `0.0` in every
frame - the largest reading the fill replaced being 2.2629 rad. Reading the same
dataset back after the fix, `bob__shoulder_pan` holds `0.9` across the episode
and `bob__elbow` traces the 2.52 rad of gravity swing the joint really performed,
agreeing with the engine's own reading to within 1e-3 rad. The driven robot's
columns are bit-identical across the two recordings.

The helper now takes the collection of robots a frame drives rather than a single
name, and both `run_multi_policy` implementations (MuJoCo and Isaac) merge its
result before their own driven columns, so driven keys still win any collision.
A bare `str` is refused rather than accepted: a string is iterable per character,
which would turn the skip into a substring test and drop a robot named `ali` as
though `alice` drove it - the very fill the helper exists to avoid.

The *action* half of the same frame is unchanged and deliberately so. No command
was issued to a robot this call does not drive, so its action columns have no
truthful value to write, and they are left where #1715 left them. The structural
survey that held each recording entry point to the shared helper covered only the
single-policy hooks; it is now derived from the backend registry
`create_simulation` itself resolves and covers `run_multi_policy` for every
backend that implements one, so Isaac's half of the fix is graded on a machine
that cannot run Isaac Sim.
