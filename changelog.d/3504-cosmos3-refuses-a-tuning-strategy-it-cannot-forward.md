### Fixed: the Cosmos3 trainer refuses a tuning strategy it cannot forward

`TrainSpec.method="lora"` asks for a rank-`r` adapter. The Cosmos3 backend
accepted it and forwarded nothing: its run is configured by the recipe TOML plus
the Hydra override list `build_overrides` writes, and that list has no adapter
entry. Measured on a launchable spec, `validate()` returned no problems for
`method="lora"` - including with `lora_r=0` - and the override list it built was
byte-identical to the one for `method="full"`, so the caller asked for an adapter,
got a full fine-tune of the whole model, and was told the run succeeded.

`validate()` now refuses any strategy other than `full` on this backend, naming
the recipe TOML (`extra["sft_toml"]`) as the surface that can express one - the
posture GR00T already takes toward a strategy it has no config field for, while
LeRobot keeps honoring the request with `--peft.method_type=LORA`. A full
fine-tune builds exactly the same four overrides as before; no new override was
invented.
