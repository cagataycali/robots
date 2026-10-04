### Fixed: `create_trainer()` names the closest trainer for a near-miss provider

`create_trainer("lerbot_local")`, `"Lerobot_Local"` or `"cosmos-3"` now raises
`No trainer registered for provider ... Did you mean: 'lerobot_local'?`, the
same hint `create_policy()` gives for the same spelling, instead of only the
full list of trainers.
