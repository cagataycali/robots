### Fixed: a facing goal is graded in either spelling

`target_heading` is one well-known goal key with two spellings - a planar
direction vector, or the same direction as an angle in radians via
`target_heading_angle` - and both land in the same `facing_direction` field of
the control signal the MotionBricks generator is handed. Only the vector was
graded. The angle went through a bare `float()`, so `target_heading_angle=nan`
produced a `facing_direction` of `[nan, nan, 0.0]` from a call that reported
success (the outcome the vector spelling's own domain exists to prevent),
`True` was read as a 1.0 rad heading, a numeric string was read as an angle,
and `inf`, `10**400` and a list escaped as an unnamed `math domain error`,
`OverflowError` or `TypeError`.

The angle is now held to `finite_number_error` - the domain the sibling
locomotion family already holds its own scalar goal kwarg to
(`WBCPolicy._validate_height`) - so both spellings refuse the same seven value
classes by name. Every angle a caller can mean resolves unchanged, and the
asymmetry that is deliberate is unaffected: a bare number is a legal angle and
an illegal direction, a two-entry sequence the reverse.
