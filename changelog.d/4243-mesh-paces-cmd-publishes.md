### Fixed: a second mesh command inside 50 ms was dropped, not answered

Every peer's transport drops a cmd arriving sooner than one period after the
previous one (`STRANDS_MESH_CMD_RATE_HZ`, 20 Hz), so an agent issuing two tool
calls at once saw the second wait out its full timeout while the peer sat idle.
`Mesh.send` and `Mesh.broadcast` now pace their publishes one period plus a
margin apart, across every target, and the wait rides the mesh's stop event.
