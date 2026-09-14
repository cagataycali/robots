### Fixed: a reward-model `type` lerobot does not ship is reported on its own

`LerobotTrainer.validate()` graded `extra["reward_model"]`'s keys even when the
`type` did not resolve. `_reward_friendly_fields` answers with the offline
fallback - SARM's three documented keys - for a type the registry does not hold,
so the preflight named the "configurable fields" of a type the same call had just
declared non-native, and refused knobs that are real once the type name is
corrected: `{"type": "sram", "num_layers": 4}` reported both the typo and
`num_layers` as unsupported, though `num_layers` is one of SARM's 27 fields. A
caller acting on that second problem deletes a legitimate knob along with the
typo.

The field check is now scoped to a type that resolved, the same way the
`annotation_mode` check in that gate is already scoped to the type declaring it.
A resolved type still refuses a key it has no field for and still names its real
per-type surface; only the unresolved case loses the extra claim.
