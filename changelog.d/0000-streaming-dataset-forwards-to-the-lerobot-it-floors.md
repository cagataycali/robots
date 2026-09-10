### Changed: `streaming_dataset` forwards every knob to the lerobot it floors

`StreamingDatasetReader.open` no longer probes `StreamingLeRobotDataset`'s
signature: the `[lerobot]` extra floors lerobot at 0.6.1, whose constructor
accepts every forwarded keyword, so the `repo_type` refusal, the
`return_uint8` downgrade warning, the unknown-kwarg filter and the
`check_delta_timestamps` fallback were dead. Knob domains, the boolean-flag
refusal and `drop_videos` are unchanged. 543 -> 296 lines.
