### Fixed: `stream_dataset(episodes=[...])` streams the episodes it was asked for

lerobot's `StreamingLeRobotDataset` stores `episodes` and never reads it again -- its
materializing sibling `LeRobotDataset` does -- so a subset request streamed, and counted,
every episode in the dataset: an episode-filtered training or eval set silently held the
episodes the caller had filtered out, with nothing reporting it.

`StreamingDatasetReader.open` (behind `strands_robots.stream_dataset` and
`Simulation.stream_dataset`) now applies the subset on iteration, so it holds for the
reader's own loop AND for a `DataLoader` built over the same instance, and `num_episodes` /
`num_frames` report the subset rather than the dataset's totals. Frames outside the subset
are still fetched and dropped, so what the subset saves is what the caller consumes, not
bandwidth.

An `episodes` request the dataset cannot satisfy is refused at `open` instead of read as an
exhausted stream: a bare index or a string (both would be consumed element-by-element), an
empty, duplicated, negative or fractional list, and an index the dataset does not hold --
that last one named against the dataset's real episode count.
