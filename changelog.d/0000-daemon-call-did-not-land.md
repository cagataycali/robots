### Fixed: a Reachy Mini daemon call that did not land is no longer a success

`reachy_transport.api` reports every HTTP and connection failure as
`{"error": ...}` rather than raising. `ReachyMiniDriver.playMove`, `listMoves`,
`wakeUp` and `sleep` nested that reply under a key of their own and returned
`{"status": "success", ...}` regardless, so a caller was told the move played,
the catalogue was read and the motors were woken by a daemon that was never
reached. Three of the four command motion, and `wakeUp` also enables torque.

The four now consult the reply through one owner,
`ReachyMiniDriver._transport_failure`, which answers the refusal envelope
`getDaemonStatus` already returned - naming the verb, the daemon's address and
the transport's cause - and drops the `move`/`moves`/`result` reading of a call
that never landed. Presence rather than truthiness decides, because an HTTP
error with an empty body puts `""` under `error`; a reply that is not a mapping
is not a transport failure, so the catalogue endpoint's JSON array still reads
as a catalogue. `getDaemonStatus` now delegates that branch instead of
repeating it, and `stopMotion` keeps raising `RuntimeError` on the documented
criterion that a caller acting on a false success from a *stop* stops nothing.
