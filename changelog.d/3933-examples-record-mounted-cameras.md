### Fixed: a demo that mounts one camera records one camera

`examples/03_record_dataset.py` and `examples/07_post_tune_any_policy.py` each
mounted a single `front` sensor and then called `start_recording` with no
`cameras=`, so both datasets also carried `observation.images.default` - the
implicit free overview view `create_world` adds, which the recorder itself warns
"is not a sensor any policy declares ... and will not match a policy's
input_features".

For `03` that was a bloated dataset (two image features for one sensor) under a
docstring calling the result training-ready. For `07` it was worse, because that
demo trains on what it recorded: the phantom view became a declared visual input
of the exported checkpoint, so the artifact the demo presents as the closed
record -> train -> deploy loop is refused by any robot that has no such camera -
a `front`-only observation raises `Robot supplies 1 camera(s) ['front'] but the
policy requires image input(s) [...]; unmatched policy keys:
['observation.images.default']`. Both demos now name the sensor they mounted;
the recorded sensor track is byte-identical either way.

`tests/test_examples_record_the_cameras_they_mount.py` grades the rule the
recorder's warning states, over the examples tree rather than a hardcoded list:
a script that mounts its own camera names its cameras when it records. A script
that mounts none is not covered - there the overview view is the only view there
is, and recording it is a choice rather than an oversight.
