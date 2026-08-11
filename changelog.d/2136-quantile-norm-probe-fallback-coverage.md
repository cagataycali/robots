### Quality: pin the quantile-normalization probe's two unreadable-registry branches

`_policy_uses_quantile_norm` is the only registry probe in the LeRobot trainer that
calls a config field's `default_factory` rather than looking a field name up, so it
is the only one with a branch for "the registry holds this policy type but its answer
cannot be read". That branch, and the sibling "the config declares no such field"
branch, resolve the same question two deliberately different ways - unknown falls back
to the documented static set, a missing field is definitively `False` - and neither ran
under test. `TestOfflineFallback` pinned the offline fallback for three of the four
gates it covers; the quantile gate was the fourth.

Adds five tests: both branches, the asymmetry between them, a tripwire recording that
no config lerobot ships omits `normalization_mapping` (so the second branch needs an
injected registry today), and the consequence at the surface a caller reads - with the
registry answer unreadable, `validate`'s quantile-stats preflight still warns that a
`molmoact2` dataset lacking `q01`/`q99` would be mis-normalized. No library behaviour
changes.
