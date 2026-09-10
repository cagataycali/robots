### Fixed: the inert-normalization diagnostic prescribes the unit half of the remedy, not just the stats

A base checkpoint's normalization is inert because its stats are keyed by the
training dataset, and the load-time diagnostic pointed callers at
`processor_overrides={'normalizer_processor': ..., 'unnormalizer_processor': ...}`.
Those stats also carry the UNITS the dataset was recorded in - an SO-arm dataset
comes through the driver's `MotorNormMode`, arm joints in servo degrees and the
gripper in `RANGE_0_100` - while a MuJoCo state is radians. Supplying the stats
alone therefore rescales nothing for a sim caller: measured against
`lerobot/smolvla_base`'s `so100.buffer.action` stats, the full so101 joint range
spans 0.07-0.15 sigma packed as radians versus 3.8-8.3 sigma packed as degrees,
so `observation.state` reaches the model as a near-constant and the policy
ignores proprioception. The warning now names the embodiment's
`state_units`/`action_units` (and `joint_mids`) beside the stats, and
`docs/policies/lerobot-local.md` shows both halves in one example.
