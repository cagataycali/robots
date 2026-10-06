### Fixed: a deploy snippet with wire auth off says what that does to the arm, and where it stops working

An acknowledged real-arm deploy snippet now explains the `STRANDS_MESH_LOCAL_DEV`
line it carries (any process reaching the robot's mesh endpoint can move the
arm) and names the way back to certificates (`STRANDS_MESH_TLS_CA` / `_CERT` /
`_KEY` + `STRANDS_MESH_ACL_FILE`). A rendered local-dev line whose
`ZENOH_CONNECT` points off this machine, without
`STRANDS_MESH_I_KNOW_THIS_IS_INSECURE` in the file, carries a note that the mesh
will refuse to start there and why, instead of the operator finding out on the
edge box. The dashboard's empty-fleet hint no longer recommends
`STRANDS_MESH_LOCAL_DEV=1` + `STRANDS_MESH_MULTICAST=true` (a combination the
mesh refuses), the mesh toast and Settings no longer call a shared LAN a place
local-dev works, and the missing-resume-key warning no longer points at the
local-dev flag.
