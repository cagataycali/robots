### Fixed: a stopped real-robot task reports how long it ran

`stop` settles the elapsed time at the moment of the stop, so its reply and
every `status` afterwards carry the real figure. Before, a task stopped from
outside read `Duration: 0.0s` beside a non-zero step count, and `status` kept
repeating `Total Duration: 0.0s` for good.
