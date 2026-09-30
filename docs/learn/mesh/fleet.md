# Fleet

At the end of this page you can list a mesh's peers, ask one for its state, hand one a policy task, fan a command out, subscribe to a peer's topics, and do it all from an agent through `robot_mesh`, approvals in place.

Continuing from the [mesh index](index.md) fence (two sims, `arm-a` and `arm-b`, `STRANDS_MESH_LOCAL_DEV=true`):

```python title="sketch"
peers = a.mesh.peers                                       # presence dicts
one = a.mesh.get_peer("arm-b", max_age_s=5.0)              # None if stale
a.mesh.send("arm-b", {"action": "state"})                  # joints, sim time
a.mesh.tell("arm-b", "stack the cubes", policy_provider="lerobot_local",
            pretrained_name_or_path="lerobot/smolvla_base", duration=10.0)
a.mesh.broadcast({"action": "status"}, timeout=5.0)        # a reply per peer
a.mesh.subscribe("arm-b-state", "strands/arm-b/state", lambda key, payload: print(payload["joints"]))
a.mesh.unsubscribe("arm-b-state")
```

## Join and discover

A peer joins when `Mesh.start()` runs: `Robot(..., mesh=True)` in the constructor, `init_mesh(obj, peer_id=)` for anything else. `peer_id` is letters, digits, `.`, `_`, `-` (128 at most, the first a letter or digit); one is generated when omitted. Presence is a 2 Hz heartbeat on `strands/<peer>/presence`; a peer silent for `STRANDS_MESH_PEER_RETENTION_S` leaves `peers`.

Every process on one host meets at the local router on `STRANDS_MESH_PORT` (7447); across machines set `ZENOH_CONNECT` to the router's endpoint, or `STRANDS_MESH_MULTICAST=true` on a network you trust.

## The command vocabulary

Every command is a JSON dict whose `action` is in `ALLOWED_ACTIONS`; `strands_robots.mesh.security.validate_command` checks it on both ends; unknown actions or keys never leave the process.

| action | does |
|---|---|
| `status` | `{'status': 'idle' or 'running', 'robots_running': [...]}`; never gated, admitted under lockout |
| `state`, `features` | joint state; observation and action schema |
| `execute` | run a policy to completion: `instruction`, `policy_provider` (required), `duration`, checkpoint as a Hub id |
| `start` | the same, backgrounded |
| `step`, `reset`, `set_joints`, `call`, `describe_tool` | step once; reset; write `target_joints`; one advertised function (`function`, `params`); the served spec |
| `stop` | halt the rollout; admitted under lockout |
| `teleop_status`, `teleop_receive`, `teleop_stop` | follow a remote input stream ([teleoperation](../hardware/teleoperation.md)) |
| `resume` | clear the e-stop lockout with the override code ([safety](safety-and-estop.md)) |

Three allowlists guard what an `execute` may name: `STRANDS_MESH_POLICY_TYPE_ALLOW` (providers), `STRANDS_MESH_POLICY_HOST_ALLOW` (a policy server's `server_address`), `STRANDS_MESH_HF_REPO_ALLOW` (Hub orgs). A local checkpoint path is refused on the wire; checkpoints travel as Hub ids.

## RPC shape

`send` writes `{"action": ...}` on `strands/<target>/cmd` with a fresh 128-bit `turn_id` and waits on `strands/<me>/response/<target>/<turn_id>`; any other peer's reply is dropped. `broadcast` writes once on `strands/broadcast` and collects until `timeout`; the sender's own envelope is dropped, so `emergency_stop` stops the local robot before broadcasting.

## From an agent

```python title="sketch"
from strands import Agent
from strands_robots import robot_mesh

agent = Agent(tools=[robot_mesh])
agent("Which robots are online? Ask arm-b to wave for two seconds with the mock policy.")
```

`robot_mesh(action, target=, instruction=, command=, policy_provider=, duration=, timeout=, name=, limit=, function=)` answers `peers`, `status`, `tell`, `send`, `ping`, `rpc`, `broadcast`, `stop`, `emergency_stop`, `subscribe`, `unsubscribe`, `watch`, `inbox`. `ping` reports whether one peer is reachable and how fast; over AWS IoT an offline peer answers in one round trip ([direct messaging](direct.md)). It needs a mesh in the process.

Six actions pause for operator approval by default: `emergency_stop`, `broadcast`, `tell`, `send`, `stop`, `rpc`. `STRANDS_MESH_HITL_ACTIONS` widens or narrows that set (an unknown token is a structured error, not a silent downgrade); add `subscribe` and `watch` where telemetry is sensitive. Fleet-wide actions say so in the prompt. Each action has a sliding-window rate limit (`emergency_stop`: 3 per minute); the refusal names the wait. `rpc` calls a device-native function on a Device Connect peer (`function=`) with bounded parameters.

`STRANDS_MESH_SUBSCRIBE_ALLOW` bounds `subscribe` and `watch`; `inbox` reads what they collected, `limit` rows at a time.

## Seeing the fleet

`strands-robots dashboard` shows the same peers, state and cameras in a browser ([dashboard](../dashboard.md)). A `reach` chip says which leg carried the heartbeat: `lan`, `iot` or `both`. A robot reached over IoT is a full card, cameras included: an S3 reference the dashboard resolves itself, or with `STRANDS_MESH_IOT_CAMERA_INLINE=1` on the robot a JPEG under the 128 KB cap, latency on the tile. A provisioned Thing that never spoke is a grey `registry` card whose `ping` sends one `status` read; the Thing answers, or the broker's 404 does.
