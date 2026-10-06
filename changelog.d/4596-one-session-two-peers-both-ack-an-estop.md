### Fixed: an e-stop counts the acknowledgement of every peer id a robot's session answers as

A `Robot(..., mesh=True)` answers on the mesh as itself (`robot-a`) and as its
robot (`robot-a__so101`), both from one Zenoh session. The operator kept one
reply per session per turn, so whichever of the two answered second was dropped
as a duplicate and the e-stop reported that peer as silent ("treat them as still
moving"), although it had stopped. Replies are now deduplicated per peer id
bound to the session, so a repeated reply from the same id is still refused.
The strict per-peer ACL template (`examples/mesh/mesh_acl_strict_per_peer.json5`)
also grants the `<peer>__so101` id its cmd and reply topics, which it had denied.
