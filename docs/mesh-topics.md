---
description: What a mesh peer publishes - the topic table, degraded probes, which robot is running, and how a peer record ages.
---

# Mesh topics and peer state

Every peer on the [multi-robot mesh](mesh.md) publishes its own state and reads
its neighbours' from the topics below. This page is the wire reference; joining,
e-stop and teleop are on the [mesh page](mesh.md).

## Published topics

| Topic | Rate | Content |
|-------|------|---------|
| `strands/{peer_id}/presence` | 2 Hz | heartbeat / peer discovery |
| `strands/{peer_id}/state` | 10 Hz | joints, sim time, task status, which robots are running a policy, degraded probes |
| `strands/{peer_id}/cmd` | on demand | incoming RPC commands |
| `strands/{requester}/response/{responder}/{turn_id}` | on demand | RPC replies (turn_id correlated) |
| `strands/{peer_id}/stream` | on demand | VLA execution steps |
| `strands/{peer_id}/pose` | on demand | SE(3) from SLAM/odom/VIO |
| `strands/{peer_id}/imu` | on demand | orientation, gyro, accel |
| `strands/{peer_id}/health` | on demand | battery, CPU, memory |
| `strands/broadcast` | on demand | fan-out RPC |

Sensor topics only publish when the robot exposes the attribute. Zero cost when
unused.

A reply key is built from the request envelope, so `sender_id` (where to answer)
and `turn_id` are identifiers: `[A-Za-z0-9_.-]+`, at most 128 characters. Zenoh
routes a wildcard by intersection, so a segment holding one would address the
reply at every peer - such a command is refused whole and audited, nothing
dispatched. Omitting `sender_id` still means fire-and-forget.

A record is the reader's, not the measured peer's. A presence payload is merged
into a peer record for its capabilities (`tool_name`, `connected`, `cameras`,
...), but `peer_id`, `type`, `hostname`, `age` and `reachable` outrank it, so a
peer heartbeating `"age": 0` does not report itself fresh. A sensor record follows
the same rule - `peer_id`, and the `hand` a hand record is filed under, outrank
the provider mapping - while a provider's `t` *is* honoured, a stamp being truer
than the publish instant.

## Degraded state probes

Every section of a snapshot is optional, so an absent one is ambiguous: a robot
with no joints and a robot whose joint read just failed look identical. A failing
probe therefore names itself, and the peer publishes the diagnosis rather than
going silent:

```json
{"peer_id": "arm-a1", "t": 1755900000.123,
 "degraded": {"hw_joints": {"reason": "ConnectionError", "detail": "Port is in use!",
                            "failures": 37, "for_seconds": 3.7}}}
```

| Field | Meaning |
|---|---|
| `reason` | The exception's type name - a contended serial port is a different job from an uncalibrated arm. |
| `detail` | Its message, bounded (driver text, published 10x a second). |
| `failures` / `for_seconds` | Ticks raised and time elapsed since the fault began - one unlucky read against a standing fault. |

Categories are `hw_joints`, `task_state`, `sim_world` and `sim_joints`. An entry
clears on the tick its probe answers again, and the key is absent when nothing is
degraded.

## Which robot is running

```json
{"robots": {"arm_a": {"active": true}, "arm_b": {"active": false}}}
```

`active` means *this robot is executing a policy right now*, read from the same
in-flight population `status` answers `robots_running` from, so the topic and the
command never disagree. Background `start_policy` and blocking `run_policy` both
read `true`; the flag clears when the policy stops or expires.

A peer that cannot read that population names its robots with **no `active`
key**, answers `status` with `unknown`, and reports `sim_world` in `degraded` -
`false` would affirm "idle" on no evidence. Which robots *exist* is `sim_robots`
on the presence topic.

## Pose orientation

A 4x4 SE(3) pose provider is decomposed into `x` / `y` / `z`, a planar `theta`
and a `quat`: scalar-first `[w, x, y, z]`, unit length, sign-canonicalized to
`w >= 0` so an unchanged pose reads back identically. Both come from the same
matrix and always agree, including past 120 degrees.

## Out of contact vs gone

A peer that stops heartbeating is unreachable after `PEER_TIMEOUT` (10 s) and, by
default, deleted at that moment - which answers "was it ever here?" with "no" for
a fleet whose silence is planned, such as a rover in an RF shadow. Set
`STRANDS_MESH_PEER_RETENTION_S` to keep it in `mesh.peers` with
`reachable: false` until its silence exceeds `max(PEER_TIMEOUT, retention)`.
`STRANDS_MESH_MAX_PEERS` outranks retention: at the cap the longest-silent peer
goes first.

```python
row = robot.mesh.get_peer(peer_id, max_age_s=30.0)
if row is None:
    ...  # unknown OR older than 30 s - for this decision, the same thing
```

`max_age_s=None` (default) accepts any age. The bound must be positive and
finite: `nan` compares False for every age, failing open on the very record it
was written to refuse.

## Rejoining the mesh

`stop()` then `start()` is how a peer leaves and rejoins, keeping its identity:
the `peer_id` is unchanged and an engaged lockout stays engaged, so a network blip
is not a way to forget a stop. `stop()` waits for the sensor loops on a shared
budget; a blocking read holds one tick open past it and that loop is named at
WARNING as still able to publish once more.

Your own `subscribe()` topics do not survive - `start()` re-declares only the
built-in ones. `stop()` reports at INFO how many it dropped, and `subscribe()`
warns when it refuses, naming the topic and whether the peer is off the mesh, has
no session, or the declare failed:

```python
sim.mesh.stop()
sim.mesh.start()
name = sim.mesh.subscribe("strands/*/state", callback=on_state)
if name is None:
    ...  # the WARNING says which of the three refusals it was
```

## See also

- [Multi-robot mesh](mesh.md) - joining, e-stop, resume and teleop.
- [Mesh authentication](security/mesh.md) - the ACL postures a peer starts under.
