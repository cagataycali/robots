### Fixed: a written lerobot policy roster is graded by the direction it drifted, so a supported lerobot no longer reds the suite

The rosters naming which lerobot policy types expose a capability - the
`*_POLICY_TYPES_FALLBACK` snapshots in `strands_robots/training/lerobot.py`, and
the `validate()` paragraph of `docs/training/overview.md` - were graded for
equality against the set derived from the installed lerobot's config registry.
The manifest admits a range, `lerobot>=0.6.1,<0.7.0`, and lerobot moves a
capability inside it: `use_relative_actions` was added to `VLAJEPAConfig` after
v0.6.1, so the floor accepts four policy types for relative actions and the next
release accepts five. No written roster passes at both ends - the four fail on
the newer lerobot, and naming the fifth fails on the floor, which is the version
the lockfile resolves - so an install of a supported lerobot left two graders red
on a clean checkout with nothing to edit that would fix it.

The two directions are now graded apart, because their consequences differ. A
roster naming a type the installed lerobot has no such field on is a failure at
every version: offline that gate accepts a run lerobot will not honour. A roster
lagging a lerobot newer than the declared floor is drift a frozen roster cannot
avoid, so it is reported - naming the types, the installed version and the floor
- and becomes a failure the moment the floor is raised onto that release. The
floor is read from the manifest rather than restated, so raising it re-grades the
rosters. The Training page now says the rosters track the supported floor and
that the gate reads the reader's own lerobot.
