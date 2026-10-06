### Fixed: the README's Mesh row names the switch that starts the mesh on one machine

Copied as written, `Robot("so100", mesh=True)` under the default posture (mTLS,
no access list) refuses to start, and `robot.mesh.tell(...)` returns
`{'status': 'error', 'error': 'mesh not running'}`. The row now names
`STRANDS_MESH_LOCAL_DEV=true` for one machine and mTLS plus an ACL across hosts,
the same postures `docs/learn/mesh/index.md` lists.
