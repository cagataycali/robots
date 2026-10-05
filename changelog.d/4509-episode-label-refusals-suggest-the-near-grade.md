### Fixed: episode-label refusals name the closest grade or failure tag

`annotate_episode`, `filter_episodes` and `measure_agreement` now end a
refusal of a near-miss `quality`, `min_quality` or `failure_mode` with the
same `Did you mean: 'Medium' -> 'medium'?` clause the keyword refusals use
(`strands_robots.utils.did_you_mean`). A value close to no word in the
vocabulary keeps the message it had.
