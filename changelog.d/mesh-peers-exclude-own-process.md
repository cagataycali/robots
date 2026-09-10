### Fixed:
- `Mesh.peers` no longer lists the child peers this process spawned for its own robots (`<peer_id>__<robot_name>`), and `robot_mesh(action="peers")` no longer counts this process's own peers as "remote". A lone `Robot("so100", mesh=True)` used to show one discovered peer and report `2 local, 2 remote` with no other process on the network.
