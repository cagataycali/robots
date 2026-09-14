### Fixed: the offline policy-type snapshot names every type lerobot registers

`LerobotTrainer.validate` resolves `extra['policy_type']` against lerobot's
live `PreTrainedConfig` registry, and falls back to a static snapshot in
`strands_robots.training.lerobot` when lerobot is not importable - the case
where training cannot run but preflight should still say something useful.
That snapshot had fallen nine types behind the installed lerobot
(`eo1`, `evo1`, `fastwam`, `gaussian_actor`, `lingbot_va`, `molmoact2`,
`multi_task_dit`, `vla_jepa`, `wall_x`), so an offline preflight told a caller that a policy lerobot
ships "is not LeRobot-native" and listed a ten-name set as the alternatives.

For `molmoact2` the same `validate()` call contradicted itself: it denied the
type was a LeRobot policy and, from the quantile-normalization gate whose own
snapshot does name `molmoact2`, prescribed lerobot's
`augment_dataset_quantile_stats` script for that same type. Two offline gates in
one module disagreed about whether the policy exists.

Nothing graded either snapshot's content. Of the module's eight offline
snapshots, two were compared against the installed lerobot
(`_LEROBOT_CODEBASE_VERSION_FALLBACK`, `_SAMPLE_WEIGHTING_KEYS_FALLBACK`) and
the rest were either unchecked or checked against themselves - the existing
offline assertion is `_lerobot_policy_types() == set(_LEROBOT_POLICY_TYPES_FALLBACK)`,
which reads back the very constant it grades and so holds for any content. Both
drifted snapshots sat in that ungraded group.

Both snapshots are refreshed to the installed lerobot, and the module now
carries two invariants that are pinned rather than asserted about themselves:
each policy-type snapshot equals the live registry answer for the same question,
and every per-capability snapshot is a subset of the native-type snapshot. The
first fails when a lerobot release moves a policy type; the second needs no
lerobot at all, because a capability snapshot names LeRobot-native types by
definition, so a member the native snapshot omits is a contradiction inside the
module. A pin at the surface a caller reads drives an offline `validate()` over
every type the capability gates grade and refuses any "not LeRobot-native"
verdict among them. A name lerobot does not register is still rejected.
