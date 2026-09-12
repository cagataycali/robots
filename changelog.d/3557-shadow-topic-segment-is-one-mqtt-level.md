### Fixed: a shadow-topic segment that is not a single MQTT topic level is refused

`init_mesh` refuses `/`, `+` and `#` in a `peer_id` because "these break MQTT
topic structure and AWS Thing-name rules", but the Device Shadow mirror - the
one surface that puts that identifier on a topic - interpolated it and
`shadow_name` raw into the reserved 7-level
`$aws/things/{thing}/shadow/name/{shadow}/update`. A slash smuggled an extra
level, so the topic named a different Thing and no longer matched the reserved
pattern the broker accepts; `+` and `#` are wildcards a publish topic may not
carry; an empty name left an empty level. None of it surfaced, because
`ShadowMirror.update` is best-effort and swallows what the transport says - so
the named shadow that late-joining operators read for fleet state was simply
never written, and every call reported nothing wrong. `shadow_update_topic`,
`shadow_get_topic` and `ShadowMirror` now hold both names to the shared
mesh-identifier domain under their own names. That domain is a superset of what
`init_mesh` admits, so no identifier a live mesh accepted - a dotted `peer_id`
included - is refused here.
