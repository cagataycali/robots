### Fixed: the dashboard files a mesh sample under the peer that published it, not the peer the body names

The dashboard bridge subscribes to every peer's presence, state, stream, camera
and sensor topics with one wildcard each and took the peer identity out of the
JSON body, so any peer on the mesh could rewrite another robot's record. A
presence body saying `robot_type: sim` for a real arm made `peer_is_physical`
answer "sim" and a task started on real hardware with nobody asked (f001,
CWE-807); the same read let a peer replace another robot's camera tile, joint
readout and sensor slots and mint fleet entries for robots that do not exist
(f008, CWE-345). Identity now comes from the key expression's peer segment, the
one part of a sample mTLS and the ACL bind to the publisher: a body naming a
different peer is dropped and written to the activity trail as
`identity_mismatch`, a sample with no usable key is dropped rather than trusted,
`hw` stays sticky for the life of a peer record, presence carries the SDK's own
timestamp freshness check, and telemetry from a peer that never announced itself
mints no entry. `peer_is_physical` believes a wire sim claim only when this
dashboard launched the peer in sim mode; an uncorroborated claim answers
physical, so a forged one costs a refusal, not a rollout.
