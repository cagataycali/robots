### Fixed: a safety audit record that can never be written is announced rather than logged at DEBUG

`SensorLoopsMixin.publish_safety_event` sends one event two ways - to the mesh
wire and to the local audit log - and its own docstring records that the
`severity` argument "reaches the audit record only": the wire copy is uniformly
`info` (issue #272), so the audit record is the only copy that carries what
actually happened.

`session._report_unencodable_payload` states the rule both halves are held to. A
transport's fire-and-forget tolerance is scoped to a TRANSIENT failure - "a
closed session, a dropped broker, a socket-level write - which the next tick
retries" - and a loss that no retry can undo is reported at ERROR instead,
"because reporting it at DEBUG left the two halves of one call disagreeing". That
raised the wire half. The audit half still reported its own loss at DEBUG, below
the default level, so an operator saw nothing at all when the only surviving copy
of a `critical` event failed to reach the disk.

An audit write is permanent in the stronger sense: a safety event is published
once, at one lockout transition, so there is no later tick to retry it. The
failure is now reported at ERROR and names the event type and the real severity,
because the wire copy names neither. Everything else about the call is unchanged
- it still does not raise, the wire copy still goes out with its uniform `info`
severity, and the caller's payload is still left unedited.
