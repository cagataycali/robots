### Fixed: the LeRobot install line can record, and a recorded episode replays

The one LeRobot install line `strands_robots` prints now carries LeRobot's `dataset` extra, so after following
it `record=` works (before, it installed `[remote,feetech]` and `record=` then asked for a second install).
`Policy("<dataset dir>[?episode=n]")` replays one recorded episode: the `action` column at the dataset's fps, in
the units `meta/units.json` names. A local directory that is neither a checkpoint nor a dataset is refused by
name instead of being sent to the Hugging Face Hub as a repo id.
