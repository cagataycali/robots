### Fixed: the Training overview names the relative-action roster the gate accepts

`extra["relative_actions"]` is gated on whether a policy's lerobot config
declares `use_relative_actions`. The Training overview's copy of that roster had
been stale since it was written: it named `pi0` / `pi05` / `pi0_fast` while the
gate had already accepted `groot`, so the page denied a combination `validate()`
allowed. A hand-written roster that no test reads goes stale on the commit that
widens the gate, so the page's rosters - relative actions and quantile
normalization - are now graded against the gates that derive them, and the two
prose copies of the set inside `training/lerobot.py` (a comment one line above
the frozenset, and the `_relative_actions` docstring) point at the live
discovery instead of re-listing it.

The offline fallback itself is unchanged. `vla_jepa` declares the field only on
lerobot's unreleased `main`; on the 0.6.1 release the lockfile resolves,
`VLAJEPAConfig` has no `use_relative_actions`, so admitting it offline would
pass a preflight for a run that then either fails on the unknown flag or
silently no-ops - the failure class `validate()` exists to refuse. The roster
widens when a release carrying the field ships and the pin moves to it, at which
point `test_capability_snapshot_matches_the_live_registry` fails in the other
direction and names the addition.
