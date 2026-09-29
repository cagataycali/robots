### Added: the dashboard agent sees and drives the fleet

The dashboard's chat agent held eight tools, all about the simulation
sessions inside the dashboard process, so "which robots are online?" had no
answer and a robot started from the Devices panel was invisible to it. The
console now takes the server's mesh bridge and device manager: `fleet` lists
every robot on the mesh with its kind, freshness, joints and task (stale peers
are listed and marked, never hidden); `spawn_robot` starts a registry robot in
simulation as a mesh peer and waits for its presence, so the card is on the
dashboard when the tool returns; `despawn_robot` removes one; and every
tool-worthy peer is a native tool named after it, built by
`peer_tools.build_peer_tools`, which existed and was called by nothing. Motion
verbs on a real arm go through the existing human gate
(`agent_hitl.MotionInterruptHook`, rows derived from the built proxies). When
the fleet signature changes between turns the agent is rebuilt with the new
tools and the conversation carried over; `GET /api/agent` now reports the tool
names it will hold.
