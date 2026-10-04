### Fixed: a refused `start_recording` no longer points readers at a dataset it never wrote

`start_recording` recorded where the dataset would live before any of its refusals ran, so an unknown camera, an unrenderable camera column or a failed recorder init left `verify_dataset_episodes`, `stop_recording(bucket=...)` and replay looking at a directory that did not exist. The target is now remembered only once a recorder is armed: a refused call leaves the session's last dataset as it was, or none.
