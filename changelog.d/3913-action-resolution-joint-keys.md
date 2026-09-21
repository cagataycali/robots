### Fixed: a rollout keyed by driven joint names is unmeasured coverage, not an undriven robot

`send_action` resolves an action-dict key that names either an actuator or a
driven joint, but `PolicyRunner`'s per-actuator credit matched the key spellings
against `robot_action_keys`, so a rollout keyed by joint names credited no
actuator: `action_resolution_rate` read `0.0` for every one and
`partial_action_failure_rate` read `1.0` -- the signature of a robot that never
moved -- beside a per-step `action_resolution` of `"full"` with no unresolved
key, and with nothing in the result text, whose note only fires below `1.0`.
Which actuator such a key drove is known to the backend's resolver and not to
that loop, so the step is now resolution-unknown for per-actuator purposes and
is left out of the denominator, the rule already applied to a coarse backend
answer. Actuator-keyed and numeric-vector crediting are unchanged.
