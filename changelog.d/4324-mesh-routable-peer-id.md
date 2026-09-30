### Fixed: a peer id becomes a mesh command target only if it passes the inbound charset rule

`Mesh.send` refuses a `target` that `validate_mesh_identifier` would refuse (wildcards, slashes, whitespace, over-long) before it publishes, `Mesh._on_presence` drops a presence whose `robot_id` fails the same rule before it enters the peer registry, and the dashboard `MeshBridge` admits only routable ids into its peer table. A peer announcing itself as `*` or `**` could previously turn a command aimed at one robot into `strands/**/cmd`, a fleet-wide one (f012).
