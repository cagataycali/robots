### Fixed: `DatasetRecorder.add_frame` refuses a missing column instead of recording 0.0

A declared state name absent from the observation, or a declared action name
absent from the action when no `required_action_keys` scope is given (the
direct-API default), used to be written as `0.0` with no warning - a dataset
that `LeRobotDataset` loaded, `verify-dataset` passed and a policy trained on.
Both now raise `ValueError` naming the columns. The backends' recording hooks,
which pass the scoped set for shared scenes, are unchanged.
`docs/recording.md`'s direct-API example names the so100 sim's real keys.
