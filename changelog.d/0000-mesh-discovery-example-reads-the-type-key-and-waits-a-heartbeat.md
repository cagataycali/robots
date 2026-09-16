### Fixed: the mesh discovery example shows a peer's type and can see a second process

`examples/04_mesh_peer_discovery.py` printed `type=?` for every peer: it read
`peer_type` off each row, and `PeerInfo.to_dict()` writes the kind under
`type`. It also read the registry straight after `Robot()` returned, before
another process's heartbeat (`HEARTBEAT_HZ = 2.0`) could land, so the only row
it could ever list was this process's own child robot - reported as a
discovered peer.

The example now waits `3 / HEARTBEAT_HZ` s, reads `type` and `age`, and prints
this process's own robots separately from peers it discovered, with a hint to
start a second copy when there are none. A test ties the keys the example
reads to the keys a peer row carries, so the two cannot drift again.
