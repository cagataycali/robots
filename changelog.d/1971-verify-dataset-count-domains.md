### Fixed: the dataset-integrity gate refuses an episode-count threshold it cannot honor

`verify_dataset(root, expected=..., min_frames=...)` - the function behind
`strands-robots verify-dataset` - guarded `expected` with its own comparison and
left `min_frames`, the sibling count in the same signature, unchecked. Because the
per-episode length check runs only when `min_frames > 0`, a value that fails that
comparison did not fail loudly, it switched the check off: on a dataset holding a
zero-length episode - the corruption class the check exists to detect - a
`min_frames` of `-5`, `False` or `nan` returned `status="success"` and the CLI
exited `0`. A `"2"`, `None` or `[2]` escaped as a bare `TypeError` past a checker
documented to always produce a report, and a fractional threshold reported a frame
count no episode can have.

Both counts now share `utils.non_negative_count_error`, which keeps `0`
first-class - `min_frames=0` remains the documented way to skip the length check,
`expected=0` still asks that a dataset be empty - and rejects `bool`, closing a
hole the two hand-rolled copies shared: as an `int` subclass, `True` satisfied
`isinstance(value, int) and value >= 0` and became a silent threshold, or a silent
episode count, of one. `SimEngine.verify_dataset_episodes` routes through the same
helper, so neither surface accepts an episode count the other refuses. Refusals
are reported in the report's `problems` (exit code `1`) rather than raised,
matching how the checker already reports a corrupt dataset.
