### Fixed: the record page names Isaac as a `start_cameras_recording` backend

`docs/learn/data/record.md` scoped `start_cameras_recording()` to `[sim-mujoco]`
alone, though Isaac ships it with the same `{name}__{camera}.mp4` naming. The row
now names both backends and how each captures (MuJoCo on wall time, Isaac through
a per-step `on_frame` hook), and `docs/learn/hardware/cameras.md` says the verb
writes MP4s rather than a dataset.
