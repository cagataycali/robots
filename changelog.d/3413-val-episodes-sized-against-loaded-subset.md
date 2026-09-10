### Fixed: `val_episodes` reserves the requested count when an episode subset is loaded

`val_episodes` is delivered to lerobot as one `dataset.eval_split` FRACTION, and
lerobot multiplies that fraction by the episodes the dataset was BUILT from --
the subset left by `dataset.episodes` / `dataset.exclude_episodes` -- not by the
`total_episodes` its header declares. Both surfaces divided by the header count,
so the documented `filter_episodes` recipe reserved fewer episodes than asked:
on a 30-episode dataset filtered to 15, `val_episodes=3` held out 2. The run
still logged an eval loss, so it looked correct.

Both the `lerobot_local` trainer and the `lerobot_train` tool now size the split
against the episodes the run loads, through one shared
`utils.effective_episode_count` that delegates to lerobot's own
`resolve_episode_indices` (so an index outside the dataset shrinks the subset
here exactly as it will there). A holdout that no longer fits the subset is
refused before the run starts, naming the subset size and the dataset size.
