### Fixed: a safety event keeps the severity it was raised with

`SensorLoopsMixin.publish_safety_event` sends one event two ways and built the audit copy
by spreading the caller's payload over the parameter, so a payload carrying a `severity`
field replaced it -- with no refusal, and nothing logged:

| raised with | payload said | audit record said |
|---|---|---|
| `severity="critical"` | (no `severity` field) | `critical` |
| `severity="critical"` | `severity: "info"` | `info` |
| `severity="critical"` | `severity: 0` | `0` |
| `severity="warning"` (default) | `severity: "info"` | `info` |

What makes the replacement consequential rather than cosmetic is that there is no second
copy to compare against. The wire copy carries a uniformly `"info"` severity on purpose:
issue #272 removed per-branch severity from `strands/+/safety/event` so a subscriber could
not read it as a content-channel oracle for a rejection reason, and the method's own
comment records the consequence -- "the real severity is preserved only in the local audit
record below". A critical stop could therefore be audited as informational with nothing
anywhere disagreeing.

The precedence is not a new rule; this mixin already applies it eight times. `_read_pose`,
`_read_imu`, `_read_odom`, `_read_lidar_summary`, `_read_lidar_state`, `_read_hands`,
`_read_map_info` and `_read_health` merge a provider mapping and then re-assert the keys
this process decided, through `_stamp_local_keys`, whose docstring names the hazard --
"merged last, a provider mapping carrying one of those seeded names replaces the local
reading" -- and cites `PeerInfo.to_dict`, which spreads the peer's own payload *first* so
the locally decided keys win. `publish_safety_event` built the ninth record and was the one
that spread last. It now spreads first, the same precedence by the same reasoning.

The helper itself is deliberately not reused here: it also stamps `peer_id`, and
`log_safety_event` already carries the peer as a field of its own envelope, so routing this
record through it would change the record's shape rather than only its precedence.

A `payload` entry named `severity` is discarded from the audit copy rather than preserved
under another name, matching how `PeerInfo.to_dict` resolves the same collision; the wire
copy is untouched, so a caller who puts a severity in the payload still sees it there. The
rule is now derived from the shipped class -- no method of the mixin may place an explicit
key ahead of a `**` spread, and a method that merges with `update` must re-assert its own
keys -- so a tenth record builder added later is held to it instead of inheriting an
exemption by omission.
