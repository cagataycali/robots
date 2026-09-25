### Fixed: the getting-started notebooks record the camera they mount

`examples/notebooks/02_record_and_stream.ipynb` and
`03_record_train_deploy.ipynb` called `start_recording` without `cameras=`, so
both also recorded the implicit `default` overview view beside the `front`
camera they mount - 374.2 KB instead of 217.3 KB for 60 frames, and a schema
the streaming cell printed to the reader as
`['observation.images.default', 'observation.images.front']`. Notebook 3 then
trained on that dataset, so the checkpoint its last cell loads declared
`observation.images.default` and refused a `front`-only observation with
`unmatched policy keys: ['observation.images.default']` - an artifact no robot
outside a simulator can run. Both notebooks now scope the recording to the
sensor they mount, and the rule that grades this for example scripts reads a
notebook's code cells too.
