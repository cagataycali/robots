### Fixed: a dashboard deploy snippet for a real arm follows the spawn route's mesh auth gate

`render_snippet` for `mode="real"` now refuses with the same text the spawn
route uses when the dashboard runs with mesh wire auth off
(`STRANDS_MESH_LOCAL_DEV` or `STRANDS_MESH_AUTH_MODE=none`) and the operator has
not set `STRANDS_DASH_REAL_SPAWN_WITHOUT_MESH_AUTH`. When they have, the
local-dev line is rendered under a warning comment. Snippets no longer carry a
baked-in `STRANDS_MESH_MULTICAST=true` or `STRANDS_MESH_LOCAL_DEV=1`; they carry
the dashboard's live values only. `STRANDS_MESH_LOCAL_DEV` still defaults the
auth mode to `none`, but it counts as the insecure acknowledgement only while
the mesh stays on loopback: with multicast on or a non-loopback
`ZENOH_CONNECT` / `ZENOH_LISTEN` endpoint, `STRANDS_MESH_I_KNOW_THIS_IS_INSECURE=1`
is needed as well. The permissive-ACL refusal no longer suggests the local-dev
flag as a way out.
