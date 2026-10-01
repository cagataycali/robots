### Added: RL trainers keep a training history (`metrics.jsonl`)

`BaseRLAlgo.train` and the FastSAC / FastTD3 loops overwrote their metrics every
iteration, logged nothing per iteration and wrote no scalars, so `TrainResult.metrics`
held only the final iteration and whether a PPO / FastSAC / FastTD3 run learned could
not be judged once it ended (the lerobot and isaaclab trainers both leave a curve).
Each iteration is now appended to `<output_dir>/metrics.jsonl` (flushed per line, so a
run that dies still leaves its curve; non-finite values are `null`) and logged at INFO
every `log_interval` iterations; `TrainResult.metrics` adds `metrics_path` and
`iterations_recorded`. A 24-iteration PPO reach run records its return rising from
-8.0 to -2.0.
