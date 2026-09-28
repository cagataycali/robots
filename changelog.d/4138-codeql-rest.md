### Fixed: the CodeQL alerts in the dataset recorder, assets, registry and policies

`resolve_dataset_dir` now contains an `owner/name` id to `$HF_LEROBOT_HOME`: an id whose
segments would leave the home (`owner/../../etc`) is refused with one fixed sentence instead
of resolving, and `create(overwrite=True)` removing, a directory outside it. Explicit `root=`
and ids that are themselves paths keep their documented meaning. Every log line that quotes a
caller-supplied dataset id, robot name or directory renders it through the new
`strands_robots.utils.log_safe`, so a line break inside the value cannot forge a log entry.
The silent `except` clauses in the GR00T and LeRobot policy probes now say why the exception is
expected, or log it at debug level with the traceback where it hid a real gap.
