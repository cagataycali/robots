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
On Zenoh the dashboard's signed safety rail now sends as that peer id, not
`<dashboard>-safety`, and its approvals are deposited for the id the rail
sends as, so an operator's yes in the dashboard is spent by the command it
approved (on an IoT leg both are the Thing name).

A signed presence that verifies also records the session its leader published
from, because an approved `teleop_receive` stream is bound to that session
(teleop frames are not signed; the session is a hint, not proof), so a leader
the certificate admitted can open the stream it was approved for. The
once-per-spelling warning about a bare allowlist entry is logged by
`remote_motion_refusal` after a command is admitted, from the operator's
actual value; the allowlist matcher itself is pure, so the gate's remedy
probes no longer log or spend the warning. Posture for a mixed fleet under
`auto`: a peer whose leaf no certificate in the `STRANDS_MESH_TLS_CA` bundle
issued directly (verification is direct issuance against each certificate in
the bundle) is treated as unsigned, so an AWS-generated IoT device certificate
beside an mTLS fleet is dropped from the roster and its replies and motion
commands refused, while stop and read stay available; a fleet whose IoT
certificates come from its own registered CA with the Thing name as CN adds
that CA to the bundle, and a fleet that cannot runs `=0`.
