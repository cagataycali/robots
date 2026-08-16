### Fixed: the all-missing state-key diagnostic names a cause it checked

When none of the configured `robot_state_keys` appear in the observation,
`lerobot_local` ended its message with one fixed explanation - that generic
auto-generated keys (`joint_0..joint_N`) had been paired with a robot reporting
named joints - regardless of what the caller had actually configured. It is true
only for the case it describes, so it mis-described the cause for the callers who
most need it right: someone who configured the sim `so101` embodiment (`'1'..'6'`)
against a real SO arm was told the keys they had just chosen deliberately were
placeholders the library invented.

The cause is now read from the configured keys. A caller who configured a named
set is told those keys describe a different robot than the one reporting the
observation; the generic explanation is kept, unchanged, for the keys it was
written for. Recognition matches the loader's exact output - the consecutive
zero-based run `joint_0..joint_{n-1}` - rather than a `joint_` prefix, because
real robots also have `joint_`-prefixed joint names: the shipped `kinova_gen3`
configuration is `joint_1..joint_7`, and a prefix test called those placeholders
too.

No behaviour change: the resolved ordering, the registry-checked remedy, the
`generic_state_keys_used` telemetry, the warn-once fallback and the
`strict_keys=True` raise are all as before.
