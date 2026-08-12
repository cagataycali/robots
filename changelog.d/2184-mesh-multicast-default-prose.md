### Docs: mesh prose said multicast discovery was on by default; the config ships it off

`_zenoh_config.scouting_block()` emits `scouting/multicast/enabled=false` and
`scouting/gossip/enabled=true`, so out of the box discovery is gossip plus
explicit `ZENOH_CONNECT` endpoints and LAN multicast is an opt-in
(`STRANDS_MESH_MULTICAST=true`). Four prose sites still presented multicast as
the mechanism, which sent an operator looking for a cross-host peer that was
never going to appear: the README mesh section, `mesh.session`'s module
docstring (whose own `_build_config` docstring already said "gossip on,
multicast off by default"), `mesh.iot.camera_offload`'s description of the
Zenoh publish path, and the lerobot architecture diagram's "(default)" label.
All four now name gossip, the explicit-endpoint requirement and the opt-in
flag, and `session`'s environment-variable list documents
`STRANDS_MESH_MULTICAST`, which the README environment-variable matrix now
carries a row for. A new guard reads the shipped default from
`scouting_block()` and refuses any mesh, README or diagram prose block that
names multicast without marking it opt-in, so the two cannot drift apart
again. No behaviour change.
