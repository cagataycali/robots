### Changed: `read_dataset_episode_indices` moves to `strands_robots.verify_dataset`

The parquet ground-truth reader every episode-count check uses
(`verify_dataset`, `SimEngine.verify_dataset_episodes`, `episode_judge`) lived
in `dataset_recorder`, a module it never used and that never used it. It now
sits next to its only purpose. Import it from `strands_robots.verify_dataset`;
behaviour is unchanged.
