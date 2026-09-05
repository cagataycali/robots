### Added: peer retention - out of contact is not gone

`prune_peers` deletes a silent peer at `max(PEER_TIMEOUT,
STRANDS_MESH_PEER_RETENTION_S)` instead of unconditionally at the timeout.
Retention off (the default) is byte-identical to before; with it set, a peer
whose silence is planned - a satellite between ground-station passes, a rover
in an RF shadow - stays in the registry as a row a fleet view can render and a
dispatcher can decline to fail over, instead of being erased as if it never
existed. Every peer row now carries `reachable`, derived locally from the same
monotonic heartbeat reading as `age` and subject to the same collision rule: a
presence payload cannot claim it, because it is the field a failover trigger
reads and a peer must not answer that about itself. An unusable retention
spelling (`nan`, `inf`, negative) falls back to off with one warning per
spelling - read permissively it would mean no peer is ever pruned, leaving the
eviction cap as the registry's only bound, reached silently. The cap still
outranks retention: at the cap the longest-silent peer is evicted first.
