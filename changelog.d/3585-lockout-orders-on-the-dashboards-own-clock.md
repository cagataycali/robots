### Fixed: a lockout badge is decided on the clock that stamped the observations

`Lockout.since` is whatever the sending peer put in the safety envelope's `t` -
its own wall clock - but `resolve_peer`'s `first_seen` and `proof_at` are stamped
by the process reading it, and both were ordered against `since`. The mesh admits
an e-stop up to `STRANDS_MESH_RESUME_FRESHNESS_S` (60s) behind the receiver
before any cross-peer skew, so the two clocks disagreeing is the normal case, and
both misreadings ran the unsafe way: a peer that was already on the mesh when the
stop was published reported "this peer appeared after the fleet e-stop, so it may
never have received it", and a command accepted before the stop was even known
counted as proof the peer's lockout was not engaged - the healthy-looking badge
on a locked-out arm this module exists to prevent.

`Lockout` now carries `arrived`, the instant this process learned of the event on
its own clock, alongside `since`, the reported instant kept for display. Every
ordering question is decided on `arrived`; `as_fields` logs both, so a skew that
produced a surprising verdict is visible rather than having to be inferred. A
lockout assembled by hand rather than folded from an event has no local stamp and
still falls back to `since`. This is the rule the mesh already applies to every
duration it decides locally - `age` is the reader's reading, not the measured
peer's to report - now applied to the one stamp that had escaped it.
