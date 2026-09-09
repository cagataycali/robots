### Fixed: `stream_dataset` refuses a flag that is not a boolean, instead of inverting it

`StreamingDatasetReader.open` (and `stream_dataset` / `Simulation.stream_dataset`
through it) checked its four numeric knobs but read its five flags -
`streaming`, `shuffle`, `return_uint8`, `validate_deltas`, `drop_videos` - by
truthiness. Every non-empty string is truthy, so each flag selected the branch
the caller was opting *out* of, and a falsy non-boolean took the other branch
without being a declared spelling of it.

`drop_videos="false"` was the visible one: it *removed* the camera keys from
`delta_timestamps`, so a stream asked for with video came back proprio-only, and
when only camera keys had been requested it raised
`drop_videos=True requires delta_timestamps with at least one non-video key` -
naming a value the caller had never passed, with a remedy that leads to the
silent proprio-only stream rather than away from it. The rest were quieter:
`validate_deltas=0` skipped the delta-grid check, so an off-grid
`delta_timestamps` that `validate_deltas=True` refuses opened and streamed;
`shuffle="false"` reached LeRobot's own truthiness check and got the advancing
generator instead of the reseeded one; `return_uint8=None` streamed float32 at
~4x the bandwidth of uint8 with the warning about exactly that cost suppressed
by the same truthiness; and `streaming=0` failed inside LeRobot on `num_shards`.

All five are now held to the shared boolean domain in the same guard block as
the numeric knobs - before the lerobot import, and before `drop_videos` rewrites
`delta_timestamps` or `validate_deltas` decides whether the grid check runs - so
the refusal names the flag and the value that was actually supplied. It is the
domain the recording postures on the write side of the same dataset
(`start_recording(overwrite=...)`, `push_to_hub(private=...)`) already use.
`reader.dataloader(shuffle=...)` is unchanged: it discards the key whatever it
held, so no spelling of it can invert a branch.
