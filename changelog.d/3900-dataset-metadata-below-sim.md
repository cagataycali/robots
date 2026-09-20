### Changed: the parquet episode truth is read below every layer that reads it

`read_dataset_episode_indices` - the pure-pyarrow read of
`meta/episodes/**/*.parquet`, what a dataset actually recorded - lived inside
the `verify-dataset` checker, one layer above two of its three callers. The sim
facade's `verify_dataset_episodes` reached up for the count it certifies a
collection run with, and so did the episode judge's frame-range resolver. It now
lives in `strands_robots.dataset_metadata`, which imports `declared_count` and
nothing else internal; every caller points down and the checker leaves no name
behind. Deferred upward import inversions 12 -> 11 in
`scripts/check_import_layers.py`, whose roster line is deleted rather than
suppressed, with a new pin in `tests/test_import_layers_are_a_dag.py` asserting
the reader sits in `core`, reads nothing above it, and is read from
`sim|policies`, `app` and `tools`. Towards #3818.
