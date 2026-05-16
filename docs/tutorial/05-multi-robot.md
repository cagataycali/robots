---
description: Two Robot() instances coordinating over the Zenoh mesh — peer discovery, RPC, emergency stop.
---

# 5 — Multi-robot

`strands-robots` ships with a peer-to-peer Zenoh mesh built into the `Robot()` factory.
Every `Robot()` you create automatically becomes a peer; every other peer on the LAN
discovers it within ~1 second. This chapter walks through the patterns.

The mesh layer lives in `strands_robots/mesh/`. The optional dependency is `eclipse-zenoh`
(installed by the default `[mesh]` extra).

## TL;DR

```python
# process A
from strands_robots import Robot
sim_a = Robot("so100")
print(sim_a.mesh.peers)        # discovers sim_b shortly

# process B
sim_b = Robot("aloha")
sim_a.mesh.tell(sim_b.mesh.peer_id, "pick up the cube")
```

Both processes can be on the same machine, on the same LAN, or across networks if you
plumb a Zenoh router or Cloudflare/Tailscale tunnel.

## Setup

```bash
# zenoh ships with the default [mesh] extra — already installed if you used
# pip install strands-robots
```

Verify the install:

```python
from strands_robots.mesh import init_mesh
print(init_mesh)   # <function init_mesh ...>
```

If you see `ImportError`, your install dropped the `[mesh]` extra:
`pip install "strands-robots[mesh]"`.

## Step 1 — start two peers

In one terminal:

```python
from strands_robots import Robot

sim_a = Robot("so100")
print("peer id:", sim_a.mesh.peer_id)
print("alive: ", sim_a.mesh.alive)
```

In a second terminal:

```python
from strands_robots import Robot

sim_b = Robot("so100")
print("peers:", sim_b.mesh.peers)
# [{'peer_id': 'so100_sim-...', 'type': 'sim', 'hostname': '...', 'age': 0}, ...]
```

Discovery is automatic. The first process listens on `tcp/127.0.0.1:7447`, subsequent
processes fall back to client mode and connect through the same port.

## Step 2 — point-to-point RPC

```python
# from sim_a's process
target = sim_a.mesh.peers[0]["peer_id"]   # whatever sim_b's id is

# Ask sim_b for its task status
result = sim_a.mesh.send(target, {"action": "status"}, timeout=5.0)
print(result)
# {'type': 'response', 'responder_id': 'so100_sim-...',
#  'turn_id': 'a1b2c3', 'result': {'status': 'idle'}, 'timestamp': ...}

# Tell sim_b to execute a natural-language instruction
result = sim_a.mesh.tell(target, "pick up the cube",
                          policy_provider="mock", duration=10.0)
```

The `tell` method is shorthand for
`send(target, {"action": "execute", "instruction": "...", ...})`.

## Step 3 — broadcast

Send a command to *every* peer:

```python
results = sim_a.mesh.broadcast({"action": "status"}, timeout=2.0)
for r in results:
    print(r["responder_id"], r["result"])
```

`broadcast` returns a list of every response collected during the timeout window.

## Step 4 — emergency stop

The mesh has a built-in safety primitive: `emergency_stop()` broadcasts
`{"action": "stop"}` to every peer and writes a tamper-evident record to the audit
log:

```python
sim_a.mesh.emergency_stop()
# ~/.strands_robots/mesh_audit.jsonl now contains a JSON line for the event
```

Audit files use mode `0o600` (owner read/write only). Override the location with
`STRANDS_MESH_AUDIT_DIR`.

## Step 5 — what every peer publishes

A running mesh peer publishes a fixed set of topics. Other peers (or any Zenoh
subscriber) can read them:

```
strands/{peer_id}/presence       — 2 Hz heartbeat (peer discovery)
strands/{peer_id}/state          — 10 Hz joints / sim time / task status
strands/{peer_id}/cmd            — incoming RPC commands
strands/{peer_id}/response/{id}  — RPC replies (turn_id correlated)
strands/{peer_id}/stream         — VLA execution steps
strands/{peer_id}/pose           — SE(3) pose from SLAM/odom/VIO
strands/{peer_id}/imu            — orientation, gyro, accel
strands/{peer_id}/health         — battery, CPU, memory, temps
strands/{peer_id}/lidar/summary  — point-cloud stats
strands/{peer_id}/hand/{name}/state — end-effector state
strands/broadcast                — fan-out RPC
```

Sensor topics (`pose`, `imu`, `health`, `lidar/*`, etc.) only publish when the host
robot exposes the relevant attribute (`robot._imu`, `robot._lidar_summary`). Zero
cost when unused.

## Step 6 — let the agent coordinate

The `robot_mesh` Strands tool gives the agent the same RPC vocabulary:

```python
from strands import Agent
from strands_robots import Robot
from strands_robots.tools import robot_mesh

sim = Robot("so100")
agent = Agent(tools=[sim, robot_mesh])

agent("Find every robot on the mesh and ask each one to report its status")
agent("Tell the so100 peer to pick up the cube using mock for 10 seconds")
agent("E-STOP all peers")
```

`robot_mesh` exposes 10 actions: `peers`, `status`, `tell`, `send`, `broadcast`, `stop`,
`emergency_stop`, `subscribe`, `watch`, `inbox`. See the source in
`strands_robots/tools/robot_mesh.py` and the
[mesh README section](https://github.com/strands-labs/robots#mesh-networking).

## Step 7 — teleoperation across machines

Stream a leader arm's joint positions to a follower on another box:

```python
from strands_robots import Robot
from strands_robots.mesh import InputPublisher, InputReceiver

# Machine A — leader publishes at 50 Hz
leader = Robot("so100", mode="real")     # requires hardware
pub = InputPublisher(leader.mesh, leader.teleoperator, device_name="leader")
pub.start()

# Machine B — follower receives + applies actions
follower = Robot("so100", mode="real")   # requires hardware
rec = InputReceiver(follower.mesh, follower.robot,
                     source_peer_id=leader.mesh.peer_id)
rec.start()
```

Topic schema for `strands/{peer_id}/input/{device}`:

```json
{
    "peer_id": "leader-a1b2c3d4",
    "device": "leader",
    "method": "arm",
    "t": 1736975234.123,
    "seq": 42,
    "action": {"shoulder.pos": 1.23, "elbow.pos": -0.5, "gripper.pos": 0.0},
    "events": {"terminate_episode": false}
}
```

## Disable

| How | Scope |
|-----|-------|
| `STRANDS_MESH=false` | Process-wide kill switch |
| `Robot("so100", mesh=False)` | Per-robot opt-out |

The mesh failing for any reason (zenoh missing, port bound, network blocked) is
non-fatal — `Robot()` still returns a working sim/hardware instance with `.mesh = None`.

## Recap

- Every `Robot()` joins a Zenoh mesh automatically.
- `mesh.peers` / `mesh.send` / `mesh.broadcast` / `mesh.tell` / `mesh.emergency_stop`
  are the headline methods.
- Sensor topics auto-publish when the robot exposes the data.
- The `robot_mesh` Strands tool gives the agent the same vocabulary.
- Disable with `STRANDS_MESH=false` or `mesh=False`.

## See also

- [Tutorial 6 — Recording](06-recording.md) — record a session, distribute with the
  mesh.
- [Architecture](../architecture.md) — where the mesh sits in the module map.
- [Mesh source](https://github.com/strands-labs/robots/tree/main/strands_robots/mesh) —
  `core.py`, `session.py`, `audit.py`, `sensors.py`, `input.py`.
