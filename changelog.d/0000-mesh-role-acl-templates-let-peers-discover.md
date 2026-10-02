### Fixed: the shipped mesh ACL templates let an operator and a robot discover each other

Two mTLS peers loading `examples/mesh/mesh_acl_example.json5` or
`examples/mesh/mesh_acl_strict_per_peer.json5` never saw each other: Zenoh checks
every message on both nodes against the remote peer's subject, and the templates
granted `put` only on `ingress` and `declare_subscriber` only on `egress`, so each
message was denied on one of its two hops - presence included. Each role now
grants what it publishes (ingress `put`, egress `declare_subscriber`) and what it
reads (ingress `declare_subscriber`, egress `put`). The strict template's per-robot
keys also match the wire (`**/robot-a/state`, replies on `**/response/robot-a/*`).
A robot certificate still cannot command an operator. The two-peer acceptance
check runs under both templates as well as the permissive default.
