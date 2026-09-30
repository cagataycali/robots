### Fixed: a dashboard-spawned robot keeps the dashboard's mesh posture, and a real arm never starts without wire auth

The dashboard's three child scripts set `STRANDS_MESH_LOCAL_DEV=1` and
`STRANDS_MESH_MULTICAST=true` before they read the spawn mode, so a physical
arm started from a dashboard whose own environment named neither joined the
mesh with no mTLS, no ACL and multicast discovery on, and `LOCAL_DEV` stood in
for the insecure acknowledgement nobody gave (f005, CWE-1188 / CWE-306). The
child environment is now composed in the parent (`child_env`) and adds nothing
to the mesh posture: children inherit the dashboard's `STRANDS_MESH_LOCAL_DEV`,
`STRANDS_MESH_AUTH_MODE` and `STRANDS_MESH_MULTICAST` exactly. A `mode=real`
spawn is refused before any process exists when that posture has no wire auth,
naming `STRANDS_MESH_AUTH_MODE=mtls` as the fix and
`STRANDS_DASH_REAL_SPAWN_WITHOUT_MESH_AUTH=1` as the operator's explicit yes for
a bench they control; the spawn result names the arm's posture as `mesh_auth`.
