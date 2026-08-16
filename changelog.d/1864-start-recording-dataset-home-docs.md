### Docs: `start_recording` says where the dataset lands, and names the one override that moves it

Every backend's `start_recording` resolves the recording directory through
`resolve_dataset_dir`, so `$HF_LEROBOT_HOME` is load-bearing on all three - and
no docstring said so. A caller reading the API could not find out where a
recording went, or that the location is configurable at all:

| surface | `repo_id` | `root` |
| --- | --- | --- |
| `simulation/mujoco/recording.py` | *no entry at all* | "defaults to the LeRobot cache under `repo_id`" |
| `simulation/isaac/recording.py` | id or a local path | "overrides the repo_id cache-path resolution" |
| `simulation/newton/recording.py` | id or a local path | "overrides the repo_id cache-path resolution" |

Which cache went unnamed, so the environment variable appeared only in inline
code comments a caller never reads. The MuJoCo `Args` block documented `fps` /
`root` / `overwrite` / `vcodec` / `cameras` and omitted the one parameter that
names the dataset.

The three now point at `resolve_dataset_dir` for the precedence rules and state
the override where it is read - from LeRobot's own `HF_LEROBOT_HOME` constant, so
relocating the home moves both the recording and where `LeRobotDataset` reads it
back from. `overwrite` gets the same treatment: it cites
`DatasetRecordingMixin._prepare_dataset_target`, whose four outcomes for an
existing target (resume, clear-empty, wipe-on-overwrite, refuse-non-dataset) two
backends compressed into "wipe and recreate ... instead of appending to it" -
naming two of the four and reading as though a pre-existing empty `root` were an
error rather than the accepted case it is.

Cross-references rather than three fresh restatements, because three partial
copies of one contract is what drifted: the rules keep one prose owner, and
`TestStartRecordingDocumentsWhereTheDatasetLands` refuses a backend that stops
citing it. Ten of its twelve checks fail on the previous docstrings.

`docs/recording.md` carried the same gap plus one wrong path: its
`DatasetRecorder` snippet annotated `root=None` as `~/.strands_robots/datasets/`,
a directory no code writes a dataset to - `~/.strands_robots` is the renders,
mesh-audit and scene-cache home. A reader following it went looking for a
recording in a directory that never existed.

No behaviour change.
