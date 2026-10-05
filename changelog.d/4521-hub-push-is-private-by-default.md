### Fixed: a Hub dataset push is private unless you ask for public

`DatasetRecorder.push_to_hub` now defaults to `private=True`, matching `sync_to_bucket` and `sync_dataset_to_bucket`. `stop_recording(push_to_hub=True)` on every simulation backend takes the same `private=` (default `True`) and forwards it, where it used to publish the recorded dataset world-readable with no way to say otherwise. Pass `private=False` to publish publicly; a non-boolean `private` is refused before the episode is finalized.
