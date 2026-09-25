---
description: Two Robot() instances coordinating over the Zenoh mesh - peer discovery, RPC, emergency stop, teleop.
---

# Multi-robot mesh

<figure class="brand-figure" markdown="span">
  ![Robot peers discovering and coordinating over the Zenoh mesh](assets/mesh_network.svg){ .brand-svg }
</figure>

`Robot(name, mesh=True)` joins a Zenoh mesh. Joining is opt-in: a bare `Robot()`
leaves `robot.mesh` as `None` unless `STRANDS_MESH` is `true`/`1`/`yes`. Peers
discover each other on the LAN and can query, command and e-stop one another.

With the [`device-connect`](device-connect.md) extra installed - the recommended
networking layer - `Robot().run()` and `robot_mesh()` dispatch through it and
fall back to this mesh only when it is unavailable. Both ride on Zenoh.

The mesh refuses to start until a security posture is chosen: with no ACL,
`Robot(..., mesh=True)` logs `Mesh did NOT start` and leaves `robot.mesh.alive`
`False`. Set the developer preset below in every process that joins for localhost
work; the lab and production postures are on the
[mesh authentication](security/mesh.md) page.

```bash
export STRANDS_MESH_LOCAL_DEV=1   # both processes below; localhost only
```

```python
# process A
from strands_robots import Robot
sim_a = Robot("so100", mesh=True)
print(sim_a.mesh.peers)          # discovers sim_b within ~1 s

# process B
sim_b = Robot("aloha", mesh=True)
sim_a.mesh.tell(sim_b.mesh.peer_id, "pick up the cube",
                policy_provider="mock", duration=10.0)
```

`eclipse-zenoh>=1.6.1` (in the default install) is a floor: the safety handlers
attribute an e-stop publisher below the JSON body with `zenoh.SourceInfo`, new in
1.6.1, and an attributing receiver refuses an unattributed envelope. Upgrade a
fleet together.

## Key mesh calls

```python
# Point-to-point status query
result = sim_a.mesh.send(target_peer_id, {"action": "status"}, timeout=5.0)

# Fan-out -> responses collected within timeout
results = sim_a.mesh.broadcast({"action": "status"}, timeout=2.0)

# Safety primitive - writes a tamper-evident audit log
sim_a.mesh.emergency_stop()   # STRANDS_MESH_AUDIT_DIR overrides log location
```

## What a fleet e-stop reaches

`emergency_stop()` stops the robot in the issuing process first - a broadcast
never returns to its sender - then broadcasts `{"action": "stop"}` with no
`robot_name`, so each peer decides which of its own robots that reaches. A
hardware peer stops its task; a sim peer asks the rollouts its backend reports in
flight, else every robot the engine lists. `stop_policy` reports `was_running`, so
`stopped` names only robots whose answer was not idle:

```python
responses = sim_a.mesh.emergency_stop()
# {"ok": True,  "stopped": ["arm"], "results": {...}}          the rollout halted
# {"ok": True,  "stopped": [],      "results": {...}}          asked, none was running
# {"ok": False, "stopped": [], "not_stopped": ["arm"], ...}    a stop was refused
```

A refusal puts the peer in `peers_not_stopped`, logged at CRITICAL and carried in
the safety envelope. A backend keeping no durable rollout claim refuses rather
than affirm "nothing was running" on no evidence; bound such a rollout with
`run_policy(n_steps=...)` or `stop_when={...}` instead.

## Recovering from an emergency stop

An e-stop latches a **lockout** on every peer it reaches: that peer refuses every
command except `status`, `resume` and `stop`, and nothing clears it on a timer.

```python
sim_a.mesh.send(peer_id, {"action": "resume", "override_code": OPERATOR_CODE})
```

Configure both before you e-stop a fleet - each is only observable once you are
already locked out.

| Knob | Default | Rule |
|---|---|---|
| `STRANDS_MESH_OVERRIDE_CODE` | unset | Receivers re-verify against their own copy; unset means no remote resume at all (WARNING at startup). Same value on every peer. |
| `STRANDS_MESH_RESUME_FRESHNESS_S` | 60 | Stale past this. Trips when a receiver's clock is *ahead* of the operator. |
| `STRANDS_MESH_RESUME_FORWARD_SKEW_S` | 5 | Future-dated past this. Trips when a receiver is *behind* - the tight bound. |
| `STRANDS_MESH_RESUME_MAX_FAILS` | 5 | Wrong codes before a cooldown arms. |
| `STRANDS_MESH_RESUME_BACKOFF_S` | 30 | Cooldown length; the correct code is refused during it. |

Each bound catches one direction of skew, so widening the other does not help: a
robot 6 s behind the operator logs `refusing remote resume -- t=... in future`
for every retry until its clock is fixed. Keep fleet clocks in NTP sync; every
attempt, granted or refused, is audited.

Those bounds are the only place a *stamp* crosses a machine boundary, so
correcting a clock cannot cost you the fleet: `age`, the 10 s peer timeout,
`STRANDS_MESH_MAX_PEERS` eviction and the publish intervals are measured on
`time.monotonic()`. `Lockout` follows from that - every verdict is decided on
`arrived`, when this process learned of the stop, not on the sender's `since`.

## Published topics

What a peer publishes, and how a peer record ages:
[mesh topics and peer state](mesh-topics.md).

## Agent-driven mesh

```python
from strands import Agent
from strands_robots import robot_mesh

agent = Agent(tools=[sim_a, robot_mesh])
agent("Find every robot on the mesh and ask each one to report its status")
agent("E-STOP all peers")
```

!!! warning "A single-peer stop is graded by the answer, not by delivery"
    `robot_mesh(action="stop", target=...)` reads the envelope `Mesh.send`
    returns. A handler reporting it did not stop, a peer-level `type: error`
    (lockout, replay, authorization), a `send` precondition error, or no answer
    inside the budget (the caller's `timeout`, capped at 5 s) each make the result
    `status="error"` naming the peer, audit a failure and log at CRITICAL. A
    fleet-wide `emergency_stop` instead counts a silent peer as a gap, not a
    refusal.

!!! warning "An empty peer list is not the same as no discovery"
    `action="peers"` and `action="status"` answer from any process, hearing the
    fleet through a robot-less gateway the tool brings up on demand. When it does
    not come up, both report `no discovery ran` beside the count, name
    `STRANDS_MESH` when the kill switch is why, and audit `discovery=none`. A
    gateway that came up and heard nothing keeps the plain `0 remote` - that zero
    is a measurement.

## Mesh teleop

```python
# Machine A - leader publishes at 50 Hz  # requires hardware
leader = Robot("so100", mode="real", mesh=True)
leader_arm = Teleoperator("so101_leader", port="/dev/ttyACM1", id="leader")
leader.start_teleop_publish(teleoperator=leader_arm,
                            device_name="leader", method="arm", hz=50)

# Machine B - follower applies incoming actions  # requires hardware
follower = Robot("so100", mode="real", mesh=True)
follower.start_teleop_receive(source_peer_id=leader.mesh.peer_id,
                              device_name="leader", apply_fn=None)

leader.stop_teleop("leader")
follower.stop_teleop("leader")
```

`get_teleop_status()` inspects either side. `frames` / `frames_received`,
`errors` and `rejected` are cumulative for the life of the publisher or receiver,
while `hz_actual` measures the session running now - `start()` opens a new
window. Compare it against `hz_target` to judge the link, the totals to judge the
device.

Each frame also carries the operator's signals from `get_teleop_events()`
(`terminate_episode`, `success`, `rerecord_episode`, `is_intervention`). Reading
them is best-effort - a dead event surface never stops the joint stream - and
since the field is also `null` for a leader with no event surface, a failed read
is reported as `event_read_errors` plus a warning.

`source_peer_id` and `device_name` are single segments of
`strands/{peer_id}/input/{device_name}`, so both must be plain identifiers
(`[A-Za-z0-9_.-]+`, at most 128 chars). A wildcard or embedded `/` is refused
with a `ValidationError` rather than silently widening the stream:
`source_peer_id="**"` would apply commands from every publishing peer.

## Attach a mesh to a Simulation

`Robot(name, mode="sim", mesh=True)` is the normal path. For a `Simulation` you
built yourself, start the client and assign it:

```python
from strands_robots.mesh import init_mesh
from strands_robots.simulation import create_simulation

sim = create_simulation("mujoco")
sim.mesh = init_mesh(sim, peer_id="bench-sim")   # None when mesh is disabled
```

`Simulation(mesh=...)` takes that same started client - not a boolean opt-in, so
a truthy value with no `.stop()` (notably `mesh=True`) is rejected at
construction. `cleanup()` stops the client before tearing MuJoCo down, stepping
over a failed stop so the world and renderers are always released.

## Enable and disable

`mesh=True` opts one robot in, `mesh=False` opts one out, and
`STRANDS_MESH=true` (or `1`/`yes`) opts in process-wide for a bare `Robot()`.
`STRANDS_MESH=false` is a kill switch: it overrides `mesh=True` and refuses the
shared transport, so nothing binds the `STRANDS_MESH_PORT` listener. A mesh
failure is non-fatal - `robot.mesh` becomes `None` and the instance still works.

## Transport selection: `STRANDS_MESH_BACKEND`

An extra brings a transport's client; this variable picks which one the session
constructs. Both are needed to leave the default: the extra alone installs code
that never runs, the variable alone selects an absent client.

| Value | Transport | Extra | Notes |
|-------|-----------|-------|-------|
| `zenoh` (default) | Zenoh: the first process on a host listens on `tcp/127.0.0.1:7447` (`STRANDS_MESH_PORT`) and later ones dial it. | none | Cross-host peers need `ZENOH_CONNECT=tcp/<host>:7447`. `STRANDS_MESH_MULTICAST=true` opts into LAN scouting on `224.0.0.224:7446`, a group shared with every Zenoh application on the LAN. |
| `iot` | AWS IoT Core MQTT with X.509 mutual TLS. | `[mesh-iot]` | Needs `STRANDS_IOT_ENDPOINT`, `STRANDS_IOT_THING_NAME`, `STRANDS_IOT_CERT_DIR`. See [Security](security/mesh.md#cross-network-fleets-aws-iot-core). |
| `bridge` | Zenoh locally, mirrored to AWS IoT for fleet-wide fan-out. | `[mesh-iot]` | One publish reaches both. |

Case and whitespace are normalised, so `IOT` and `" iot "` both select `iot`. An
unrecognized value falls back to `zenoh` and is reported once per distinct
offending value: a typo keeps the mesh running and stays visible. The vocabulary
lives in `strands_robots/mesh/_backend_select.py`, read by both the session gate
and the transport factory.

## See also

- [Mesh topics and peer state](mesh-topics.md) - what a peer publishes and how a record ages.
- [Device Connect](device-connect.md) - the recommended networking layer this mesh backs.
- [AI agents](agents.md) - drive the mesh with natural language.
- [Mesh source](https://github.com/strands-labs/robots/tree/main/strands_robots/mesh) - `core.py`, `session.py`, `audit.py`, `sensors.py`, `input.py`.
