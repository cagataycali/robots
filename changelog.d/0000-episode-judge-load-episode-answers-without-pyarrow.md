### Fixed: `load_episode` answers with an envelope when the parquet reader is absent

`strands_robots.tools.episode_judge` documents that every tool returns the
`{"status", "content"}` envelope and never raises, so a judge run over a hundred
episodes reports the one episode it could not read instead of dying on it.
`load_episode` reaches the episode metadata through
`verify_dataset.read_dataset_episode_indices`, whose documented first failure is
`ImportError` for an absent `pyarrow` - it ships with the `lerobot` extra, so a
judge process that only reads datasets recorded elsewhere can lack it - and the
tool's handler caught only `(ValueError, OSError)`. The import error escaped the
tool as a traceback. `ImportError` now joins that tuple, matching the callee's
other two callers: `verify_dataset.verify_dataset` reads it in one tuple and
`SimEngine.verify_dataset_episodes` in a handler of its own. The sibling tool
`sample_frames`, which reaches parquet through this module's own frame reader,
already refused correctly.
