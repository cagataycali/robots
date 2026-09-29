# Fleet

At the end of this page you can list the peers on a mesh, ask one for its state, hand one a policy task, fan a command out to all of them, subscribe to a peer's topics, and do the same from an agent through the `robot_mesh` tool with its approvals in place.

Continuing from the [mesh index](index.md) fence (two sims, `arm-a` and `arm-b`, `STRANDS_MESH_LOCAL_DEV=true`):

```python title="sketch"
peers = a.mesh.peers                                       # presence dicts: peer_id, robot, last_seen, ...
one = a.mesh.get_peer("arm-b", max_age_s=5.0)              # None if stale
a.mesh.send("arm-b", {"action": "state"})                  # joints, sim time
a.mesh.tell("arm-b", "stack the cubes", policy_provider="lerobot_local",
            pretrained_name_or_path="lerobot/smolvla_base", duration=10.0)
a.mesh.broadcast({"action": "status"}, timeout=5.0)        # a reply per peer
a.mesh.subscribe("arm-b-state", "strands/arm-b/state", lambda key, payload: print(payload["joints"]))
a.mesh.unsubscribe("arm-b-state")
```

## Join and discover

A peer joins when its `Mesh.start()` runs: `Robot(..., mesh=True)` does it in the constructor, `init_mesh(obj, peer_id=)` does it for anything else. `peer_id` starts with a letter or digit and continues with letters, digits, `.`, `_`, `-` (128 characters at most); one is generated when omitted. Presence is a 2 Hz heartbeat on `strands/<peer>/presence`; a peer silent for `STRANDS_MESH_PEER_RETENTION_S` drops out of `peers`.

Every process on one host meets at the local router on `STRANDS_MESH_PORT` (7447). Across machines set `ZENOH_CONNECT` to the router's endpoint, or turn on `STRANDS_MESH_MULTICAST=true` on a network you trust.

## The command vocabulary

Every command is a JSON dict with an `action` from `ALLOWED_ACTIONS`, validated by `strands_robots.mesh.security.validate_command` on both the sending and receiving side. Unknown actions and unknown keys never leave the process.

| action | does |
|---|---|
| `status` | `{'status': 'idle' or 'running', 'robots_running': [...]}`; never gated, admitted under lockout |
| `state`, `features` | the joint state; the observation and action feature schema |
| `execute` | run a policy to completion: `instruction`, `policy_provider` (required, no silent default), `duration`, checkpoint as a Hub id |
| `start` | the same, in the background |
| `step`, `reset`, `set_joints`, `sim_call` | step; reset; write `target_joints`; one published simulation action (`sim_action`, `params`), simulation peers |
| `stop` | halt the rollout; admitted under lockout |
| `teleop_status`, `teleop_receive`, `teleop_stop` | follow a remote input stream ([teleoperation](../hardware/teleoperation.md)) |
| `resume` | clear the e-stop lockout with the override code ([safety](safety-and-estop.md)) |

Three allowlists guard what an `execute` may name: `STRANDS_MESH_POLICY_TYPE_ALLOW` (providers), `STRANDS_MESH_POLICY_HOST_ALLOW` (`server_address` of a policy server), `STRANDS_MESH_HF_REPO_ALLOW` (Hub orgs for checkpoints). A local checkpoint path is refused on the wire; checkpoints travel as `lerobot/...` ids.

## RPC shape

`send` writes `{"action": ...}` on `strands/<target>/cmd` with a fresh 128-bit `turn_id` and waits on `strands/<me>/response/<target>/<turn_id>`. A reply from any peer other than `target` is dropped, so a peer cannot answer for another. `broadcast` writes once on `strands/broadcast` and collects until `timeout`; the sender's own envelope is dropped on receipt, which is why `emergency_stop` stops the local robot itself before broadcasting.

## From an agent

```python title="sketch"
from strands import Agent
from strands_robots import robot_mesh

agent = Agent(tools=[robot_mesh])
agent("Which robots are online? Ask arm-b to wave for two seconds with the mock policy.")
```

`robot_mesh(action, target=, instruction=, command=, policy_provider=, duration=, timeout=, name=, limit=, function=)` answers `peers`, `status`, `tell`, `send`, `ping`, `rpc`, `broadcast`, `stop`, `emergency_stop`, `subscribe`, `unsubscribe`, `watch`, `inbox`. `ping` reports whether one peer is reachable and how fast; over AWS IoT an offline peer answers in one round trip ([direct messaging](direct.md)). It needs a mesh in the process (`Robot(mesh=True)` or a gateway).

Six actions pause for operator approval by default: `emergency_stop`, `broadcast`, `tell`, `send`, `stop`, `rpc`. `STRANDS_MESH_HITL_ACTIONS` widens or narrows that set (an unknown token is a structured error, not a silent downgrade); `subscribe` and `watch` can be added for operators who treat telemetry as sensitive. Fleet-wide actions (`emergency_stop`, `broadcast`) say so in the prompt. Each action has a sliding-window rate limit (`emergency_stop` at 3 per minute) and the refusal names the wait. `rpc` calls a device-native function on a Device Connect peer (`function=`), charset-checked with bounded parameters.

`subscribe` and `watch` are bounded by `STRANDS_MESH_SUBSCRIBE_ALLOW`; `inbox` reads what a subscription collected, `limit` rows at a time.

## Seeing the fleet

`strands-robots dashboard` shows the same peers, their state and cameras in a browser ([dashboard](../dashboard.md)).
