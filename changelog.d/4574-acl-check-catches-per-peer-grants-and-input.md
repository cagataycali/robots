### Fixed: the permissive ACL check catches per-peer and prefix command grants and the teleop input topic

The check that refuses to start the mesh under a permissive ACL tested each
`allow` + `put` grant against four sample keys, so a grant naming a real robot
(`**/neon/cmd`), a prefix glob (`scout-$*/cmd`) or the teleop input stream
(`**/input/**`) passed silently. It now asks whether the key expression can
reach any peer's `cmd`, the broadcast, `safety/*` or `<peer>/input/<device>`
for any peer name, glob or namespace, and treats a key expression it cannot
read as reaching them. Such a grant bound to a subject with no
`cert_common_names` now refuses to start unless
`STRANDS_MESH_ACCEPT_PERMISSIVE_ACL=1` acknowledges it; before, it only logged
a warning. The ACL docs now say what a live Zenoh session shows: `key_exprs`
match the key with the namespace in front (`<namespace>/strands/<peer>/cmd`),
not stripped.
