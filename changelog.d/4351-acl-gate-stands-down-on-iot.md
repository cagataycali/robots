### Fixed: a peer on the pure `iot` backend starts without `STRANDS_MESH_ACCEPT_PERMISSIVE_ACL=1`

`Mesh.start` refused a permissive Zenoh ACL under mtls even when
`STRANDS_MESH_BACKEND=iot`, where no Zenoh session opens and the AWS IoT
policy on the thing's certificate is the access-control list. A fresh
`provision_robot` setup therefore logged `Mesh did NOT start` until the
operator set the opt-in the security docs forbid in production. The gate
now proceeds on `iot` with one INFO line; `zenoh` and `bridge` keep it.
