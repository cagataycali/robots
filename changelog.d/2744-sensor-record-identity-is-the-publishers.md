### Fixed: a sensor record's identity is the publisher's, not the provider's

Every `SensorLoopsMixin._read_*` reader seeds a record with the keys this process decided,
merges the robot's provider mapping over it, and publishes the result to a topic it builds
from those same keys. Merged last, a provider mapping carrying one of the seeded names
replaced the local reading. Nine merge sites across seven readers were affected, so a
reading published to `strands/{peer_id}/...` could name a different peer inside it, and a
hand record published under one hand's name could name another:

| published to | provider mapping said | record said |
|---|---|---|
| `strands/alice/imu` | `peer_id: bob` | `peer_id: bob` |
| `strands/alice/pose` | `peer_id: bob` | `peer_id: bob` |
| `strands/alice/lidar/summary` | `peer_id: bob` | `peer_id: bob` |
| `strands/alice/hand/left/state` | `peer_id: bob`, `hand: RIGHT` | `peer_id: bob`, `hand: RIGHT` |
| `strands/alice/health` | `peer_id: bob` | `peer_id: alice` |

The precedence is not a new rule. `docs/mesh.md` already states it for the presence payload
-- the locally decided keys win a name collision with what a peer reports, because "the
`peer_id` a peer is filed under is the one its topic and certificate bind - not a field
inside the payload" -- and `PeerInfo.to_dict` implements it by spreading the peer's own
payload *first*, its docstring naming exactly what spreading it last costs: "a `peer_id`
the sender chooses is the key `Mesh.peers_by_id` and `Mesh.get_peer` look the peer up by".
The health reader was already immune for a different reason, visible as the last row above:
it lifts named fields (`battery.get("pct")`) and namespaces the rest under its own key, so
no provider name reaches the top level of its record.

The locally decided keys are now re-asserted after each merge, through one helper so the
nine sites cannot drift apart, and the readers that merge wholesale are derived from the
shipped class rather than listed -- an eighth such reader added later is held to the same
precedence instead of inheriting an exemption by omission.

Two boundaries are deliberately unchanged. `t` is not re-asserted: it is a stamp rather
than a locally computed duration, so a provider that stamps a reading when it decoded it
reports something truer than the moment the loop got round to publishing it. The
`source`/`frame` labels keep their `setdefault` seeding, which exists precisely so a
provider may name its own frame.
