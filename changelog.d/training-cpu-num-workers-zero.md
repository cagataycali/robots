### Fixed
- `LerobotTrainer` on CPU defaults `num_workers` to 0 so the in-process `train()` no longer spawns loader workers that re-import the caller's script; `extra={"num_workers": N}` still wins.
