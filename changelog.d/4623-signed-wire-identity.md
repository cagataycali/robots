### Security: mesh messages are signed by the sender's certificate; approvals name their actor

The mesh identified the sender of a presence, a reply or a command by the
Zenoh `SourceInfo` the publisher attached to its own sample. That label is one
the publisher chooses: `zenoh.SourceInfo(entity_id, sn)` accepts any id copied
off another peer's heartbeat, with or without mTLS. An admitted peer could
therefore answer a fleet emergency stop in another robot's name (the operator
saw the robot as stopped and the robot's genuine "did not stop" reply was
discarded as a duplicate), and spend a motion approval that was granted to
someone else, because `STRANDS_ROBOT_COMMAND_ALLOW` and a dashboard grant
named a verb but no sender. After ten seconds of silence any peer could also
announce itself under a trusted peer's name.

Every peer that holds a certificate now signs what it publishes
(`strands_robots.mesh.wire_identity`): presence, replies, `send` and
`broadcast` carry a `sig` block with the DER leaf, a time, a nonce and a
signature over the canonical body. The mTLS pair signs under
`STRANDS_MESH_AUTH_MODE=mtls`; the AWS IoT device certificate signs on the
`iot` and `bridge` backends. `STRANDS_MESH_REQUIRE_SIGNED_IDENTITY` decides
what a receiver requires: `auto` (default) requires a verifiable signature
exactly when the mesh runs mTLS and `STRANDS_MESH_TLS_CA` loads as a trust
root, `1` always, `0` keeps the session-id path. When required, a presence
binds its `robot_id` to the signing certificate (the CN must be the id, or its
parent for a `<peer>__<robot>` child; a second certificate claiming a live
name is dropped and audited, and silence does not open the name), a reply is
accepted once per nonce from a certificate whose CN speaks for its
`responder_id`, and a motion command is attributed to its signer. A command
also names its target inside the signed body (`target_id`: the peer, or the
broadcast sentinel), so a genuine signed command captured off one robot's
`cmd` topic and republished unchanged on another's is refused before the
motion gate and audited as `command_refused` / `signed_for_another_peer`,
instead of running there and spending that robot's own approval for its
sender. The legacy session-id path no longer accepts an unattributed reply on
a Zenoh leg of the `bridge` backend.

Approvals are scoped to the verified actor. A dashboard grant records the
peer it was given for (the dashboard's own mesh peer id for a proxy tool;
in-process for `pose_tool` / `serial_tool`) and is left unspent by any other
sender. `STRANDS_ROBOT_COMMAND_ALLOW` accepts `<verb>@<peer>` and `*@<peer>`;
a bare `<verb>` or `*` still admits every verified sender and is warned about
once, naming the scoped spelling. The Reachy Mini Device Connect RPCs that
move the head (`look`, `antennas`, `body`, `enableMotors`, `playMove`, `nod`,
`shake`, `happy`, `wakeUp`, `sleep`) now run the same operator gate as the
`execute` RPC with the caller as the actor; stopping and reading are not
gated, and every RPC on the driver is classified in one of the two sets.
`STRANDS_DASHBOARD_PEER_ID` pins the dashboard's own mesh peer id to the name
its certificate speaks for, in place of the per-start `dashboard-<host>-<hex>`.
