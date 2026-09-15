### Fixed: the mesh startup posture warnings are said once per process, not once per peer

`Mesh.start()` warned about a missing `STRANDS_MESH_OVERRIDE_CODE` (and about
`STRANDS_MESH_MULTICAST=true`) from every instance. A sim on the mesh starts
two - the sim itself and a child peer per robot - so
`examples/04_mesh_peer_discovery.py` printed the four-line e-stop banner twice,
under `example-arm-01` and `example-arm-01__so100`. Both read the same
environment; the second banner carried no new fact and doubled the noise the
example's one real output line sat under.

Each posture warning is now emitted once per process, by the first peer to
start, and the posture itself is unchanged: an unset resume code or an opted-in
multicast still warns, a set code stays silent, and the two kinds are
independent.
