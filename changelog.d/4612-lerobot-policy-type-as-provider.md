### Fixed: a LeRobot policy type passed as the provider is sent to `lerobot_local`

`create_policy("act")` (and `pi0`, `pi05`, `pi0_fast`, `smolvla`, `diffusion`,
`vqbet`, `tdmpc`, `molmoact2`) used to end in the generic "Unknown policy
provider" list, and `resolve_policy("act")` forwarded `"act"` to `lerobot_local`
as a checkpoint id. Each is now refused with one sentence naming the call that
works: `policy_provider='lerobot_local'` with `policy_type='act'` and a
`pretrained_name_or_path`. The simulation's policy tools and `create_trainer`
report the same sentence.
