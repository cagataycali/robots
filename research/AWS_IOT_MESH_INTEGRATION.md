# Native AWS IoT Integration for the strands-robots Mesh

**Status:** Research / Design — no implementation yet
**Author:** cagataycali (with DevDuck)
**Date:** 2026-05-16
**Companion PR for context:** [#101 — feat(mesh): full Zenoh mesh](https://github.com/strands-labs/robots/pull/101)
**Branch reviewed:** `cagataycali:autonomous/mesh-session`

---

## TL;DR

The mesh layer that just landed in PR #101 is, by accident of good design, **one
refactor away from speaking AWS IoT MQTT natively**.

Every payload it produces (`presence`, `state`, `cmd`, `response`, `safety/event`,
`pose`, `imu`, `health`, …) flows through a single chokepoint —
`strands_robots.mesh.session.put(key, data)` — and every subscription is a
single `session.declare_subscriber(key, handler)` call. The topic scheme is
already MQTT-safe (`strands/{peer_id}/{kind}` with one slash-separated segment
per concept), the payloads are already JSON, and the per-peer namespacing is
already shaped exactly like IoT Policy's `${iot:Connection.Thing.ThingName}`
substitution.

We do **not** want to replace Zenoh with AWS IoT. We want to make AWS IoT a
**second transport** behind the same `Mesh` API. Zenoh stays as the LAN
hot-loop substrate (sub-millisecond, multicast, peer-to-peer); AWS IoT Core
becomes the WAN bridge (mTLS authenticated, durable shadow state, rules-engine
routed, audit-trail backed).

This document is the design discussion — not the diff. It exists to make sure
we land the abstraction in the right place and don't paint ourselves into a
corner before we write a single line of `awscrt` code.

---

## Table of contents

1. [Why this is a good idea (and what AWS IoT actually buys us)](#1-why)
2. [What the mesh looks like today (verbatim, against `pr-101`)](#2-mesh-today)
3. [The architectural insight — IoT is a transport, not a replacement](#3-insight)
4. [Topic mapping — Zenoh keys to MQTT topics](#4-topic-mapping)
5. [State, identity, and the Device Shadow](#5-shadow)
6. [Auth, safety, and the E-Stop story](#6-safety)
7. [The honest costs, latencies, and footguns](#7-costs)
8. [Proposed architecture (still no implementation)](#8-architecture)
9. [`MeshTransport` protocol — the one refactor](#9-mesh-transport)
10. [Migration plan — 4 layers, each independently shippable](#10-migration)
11. [Killer demos this enables](#11-demos)
12. [Open questions / unresolved risks](#12-risks)
13. [Decision log — what we explicitly chose NOT to do](#13-non-goals)
14. [Next steps](#14-next-steps)

---

<a id="1-why"></a>

## 1. Why this is a good idea (and what AWS IoT actually buys us)

The Zenoh mesh shipped in PR #101 is excellent for the LAN hot-loop:

- **<1 ms publish latency** on a single LAN.
- **Multicast scouting** = zero-config peer discovery on the same broadcast
  domain.
- **Self-organizing**: the first process to bind `tcp/127.0.0.1:7447` becomes
  the local router; subsequent ones fall back to client mode and connect to it.
- **Ref-counted session singleton**: 100 robots in the same process share one
  Zenoh session. Lazy `import zenoh` so users without `[mesh]` extras pay
  nothing.
- **Topic-based pub/sub**: presence, state, cmd, response, broadcast, camera,
  pose, imu, odom, lidar, hand, map, safety, input, stream — all under
  `strands/{peer_id}/{topic}`.

That's exactly what you want for a leader arm publishing 50 Hz teleop input to
a follower arm one hop away. It is **not** what you want when the robot is in
Tokyo, the operator is in Seattle, the Bedrock agent is in `us-west-2`, and
the auditor is in Frankfurt. For that, AWS IoT Core gives us six things
Zenoh-on-LAN can't, and one thing it shouldn't try to:

| AWS IoT feature | Why it matters for robots | What it replaces in our stack today |
|---|---|---|
| **IoT Core MQTT broker** (mTLS) | Cross-region, cross-VPC, cross-internet pub/sub with X.509 mutual TLS. Robot at home talks to fleet ops in `us-west-2` without a VPN. | The `tcp/127.0.0.1:7447` listener — fine on LAN, doesn't traverse NATs and CGNATs. |
| **Device Shadow** (named shadows) | Authoritative cloud-side state per robot. Survives disconnect. Last-known-good queryable from Lambda / Bedrock without an MQTT subscription. | Our `presence` + `state` topics — but **persistent**, not best-effort. |
| **IoT Rules Engine** | SQL-like routing and transformation. `WHERE health.battery_pct < 10 → SNS alert`, `WHERE safety.event.severity = 'critical' → Lambda + DynamoDB`, `WHERE * → Kinesis Firehose → S3 → Athena`. | The audit log, but queryable + alerting. Today you'd grep JSONL on each robot. |
| **Greengrass v2** | Local broker on the robot host, MQTT bridge, runs Lambda components locally even when offline, HSM credential storage. | The "router-on-the-robot" story, plus offline buffering and native auth. |
| **Fleet Provisioning** | Robot ships from factory with a bootstrap claim cert and obtains its real cert + Thing name on first boot via a registration Lambda. | `peer_id = f"{base}-{uuid.uuid4().hex[:8]}"` — unauthenticated, collision-prone, unowned. |
| **IoT Jobs** | Versioned deploy of policies / firmware / data_configs to a fleet, with rollout %, canaries, status callbacks. | Nothing today. We `pip install` per host. |
| **TwinMaker / Sitewise** | Industrial digital twin and time-series storage. Better than CloudWatch for 50–200 Hz joints. | Our state stream goes nowhere persistent today. |

The thing AWS IoT shouldn't try to replace: **the LAN hot loop**. Teleop input
at 50 Hz over the public internet to a follower arm is doomed by latency
regardless of which broker you use. The bridge transport must keep
`strands/{peer}/input/{device}` strictly local.

---

<a id="2-mesh-today"></a>

## 2. What the mesh looks like today (verbatim, against `pr-101`)

### 2.1 The single chokepoint

```python
# strands_robots/mesh/session.py
def put(key: str, data: dict[str, Any]) -> None:
    """Publish a JSON payload to the mesh. Fire-and-forget. No-op if no session."""
    if _SESSION is None:
        return
    try:
        _SESSION.put(key, json.dumps(data).encode())
    except Exception as exc:
        logger.debug("Zenoh put error on %s: %s", key, exc)


def get_session() -> Any | None:
    """Acquire the shared Zenoh session (lazy, ref-counted). None if Zenoh missing."""
    ...
```

`Mesh` and `SensorLoopsMixin` and `InputPublisher` and `audit.py` all call
exactly these two entry points. There is no other path to the wire. **That's
the entire surface area we need to swap.**

### 2.2 The topic scheme

Captured from `mesh/core.py`, `mesh/sensors.py`, and `mesh/input.py`:

| Topic | Direction | Rate | Purpose |
|---|---|---|---|
| `strands/{peer}/presence` | out | 2 Hz (`HEARTBEAT_HZ`) | Capability-bearing heartbeat. Carries `robot_id`, `robot_type`, `hostname`, `tool_name`, `task_status`, `instruction`, `connected`, `hw`, `cameras`, `inputs`, `action_keys`, `world`, `sim_robots`, `topics`. |
| `strands/*/presence` | in | – | Wildcard subscription that populates the peer registry. |
| `strands/{peer}/state` | out | 10 Hz (`STATE_HZ`) | Live robot state. Carries `joints` (filtered for non-image obs), `task` (status/instruction/steps/duration), `sim_time`, `robots`. |
| `strands/{peer}/cmd` | in | event | RPC inbox. Payload: `{sender_id, turn_id, command:{...}, timestamp}`. |
| `strands/broadcast` | in | event | Fan-out RPC inbox. |
| `strands/{peer}/response/{turn_id}` | in | event | RPC reply, correlated by `turn_id`. |
| `strands/{peer}/camera/{cam}` | out | opt-in (`STRANDS_MESH_CAMERA_HZ`, default 0) | JPEG-encoded frames (base64 inside JSON). |
| `strands/{peer}/pose` | out | 10 Hz (`POSE_HZ`) | SE(3) from SLAM / odometry / VIO. Source-tagged (`provider` / `slam` / `odom`). |
| `strands/{peer}/health` | out | 0.5 Hz (`HEALTH_HZ`) | Battery, CPU, mem, disk, uptime, temps. |
| `strands/{peer}/imu` | out | 10 Hz (`IMU_HZ`) | RPY, gyro, accel. |
| `strands/{peer}/odom` | out | 10 Hz (`ODOM_HZ`) | Dead-reckoning. |
| `strands/{peer}/lidar/summary` | out | 5 Hz (`LIDAR_SUMMARY_HZ`) | Point cloud stats. |
| `strands/{peer}/lidar/state` | out | 1 Hz (`LIDAR_STATE_HZ`) | Sensor state. |
| `strands/{peer}/hand/{name}/state` | out | 50 Hz (`HAND_HZ`) | End-effector joints / force. |
| `strands/{peer}/map/info` | out | 0.2 Hz (`MAP_INFO_HZ`) | Map metadata. |
| `strands/{peer}/safety/event` | out | event | Safety events. Mirrors to `~/.strands_robots/mesh_audit.jsonl`. |
| `strands/safety/estop` | out | event | Global E-stop record (the `broadcast({"action":"stop"})` is what actually stops peers). |
| `strands/{peer}/input/{device}` | out | 50 Hz (`INPUT_HZ_DEFAULT`) | Teleop leader stream. |
| `strands/{peer}/stream` | out | per VLA step | Observation + action + instruction snapshot. |

Every payload is a JSON dict with `peer_id` and `t` (UNIX timestamp) plus
topic-specific fields. **Nothing is binary except the base64-encoded camera
frames** (which themselves live inside JSON).

### 2.3 The peer registry

```python
# session.py
@dataclass
class PeerInfo:
    peer_id: str
    peer_type: str = "robot"
    hostname: str = ""
    last_seen: float = 0.0
    caps: dict[str, Any] = field(default_factory=dict)
```

In-process dict, TTL of 10 seconds, populated by `update_peer()` from
`Mesh._on_presence`. Pruned every heartbeat tick. **Lost on process exit.**

### 2.4 The RPC mechanism

`Mesh.send(target, cmd, timeout)` and `Mesh.broadcast(cmd, timeout)`:

1. Allocate `turn_id = uuid.uuid4().hex[:8]`.
2. Register `_pending[turn_id] = threading.Event()`.
3. `put(f"strands/{target}/cmd", {sender_id, turn_id, command, timestamp})`.
4. Subscribe (declared at start) on `strands/{my_peer}/response/**` already
   maps inbound responses by `turn_id` and signals the event.
5. `event.wait(timeout)`, collect responses, return.

Correlation is **topic-based**, not header-based. It maps cleanly onto MQTT v3
without any v5 features needed, but MQTT v5's `correlationData` is a strictly
better implementation when both sides are v5-capable.

### 2.5 The audit log

`mesh/audit.py` writes append-only JSONL to `~/.strands_robots/mesh_audit.jsonl`,
mode 0o600, parent dir 0o700, per-process write lock. Every safety event hits
this file. **Today it's local-only.** That is the right primary; what we add
in the cloud is a mirror, not a replacement.

### 2.6 The agent-facing tool

`tools/robot_mesh.py` exposes `peers / status / tell / send / broadcast / stop /
emergency_stop / subscribe / unsubscribe / watch / inbox` to a Strands agent.
**This tool is transport-agnostic by construction** — it never imports Zenoh
directly. It calls `mesh.send()`, `mesh.broadcast()`, `mesh.subscribe()`. So
the agent UX stays identical when we swap the transport.

---

<a id="3-insight"></a>

## 3. The architectural insight — IoT is a transport, not a replacement

If you abstract `put()` and `declare_subscriber()` behind a `MeshTransport`
protocol, you can have:

1. **`ZenohTransport`** — current behaviour. LAN hot-loop. Default.
2. **`IotMqttTransport`** — pure cloud. mTLS to AWS IoT Core. Designed for
   robots that have no LAN peers (a single G1 in a customer's home talking
   straight to fleet ops).
3. **`BridgeTransport`** — both. Local Zenoh router for in-process and
   on-LAN peers; outbound topics matching a filter list also publish to AWS
   IoT Core; inbound from MQTT (cmd / broadcast) is replayed onto the local
   Zenoh bus so existing handlers see it. **This is the production mode for
   real fleets.**

Selection happens via env / config, **never via API**:

```bash
STRANDS_MESH_BACKEND=zenoh        # default — current behaviour
STRANDS_MESH_BACKEND=iot          # pure cloud — single-robot, no LAN peers
STRANDS_MESH_BACKEND=bridge       # production — both, with topic filter
```

The agent code (`Robot()`, `Simulation()`, `robot_mesh` tool) **doesn't care
which is selected**. That's the whole point.

---

<a id="4-topic-mapping"></a>

## 4. Topic mapping — Zenoh keys to MQTT topics

The good news: every existing Zenoh key is already a valid MQTT topic. The
bad news: not every existing topic should leave the LAN.

### 4.1 The mapping table

| Zenoh key | MQTT topic (in IoT Core) | QoS | Retain | Bridges? | Why |
|---|---|---|---|---|---|
| `strands/{peer}/presence` | `strands/{peer}/presence` | 1 | **yes** | up | Late subscribers (Lambda, Bedrock agent) need the last presence. Retained matches Zenoh's behaviour for new peers. |
| `strands/{peer}/state` | `strands/{peer}/state` | 0 | no | up (filtered) | 10 Hz × N robots is volume. Send via **IoT Basic Ingest** (free) → Kinesis → Timestream. Don't subscribe live unless debugging. |
| `strands/{peer}/cmd` | `strands/{peer}/cmd` | 1 | no | both | RPC must arrive. Bridges down (cloud → robot) for remote ops. |
| `strands/broadcast` | `strands/broadcast` | 1 | no | both | Same. |
| `strands/{peer}/response/{turn}` | `strands/{peer}/response/{turn}` | 1 | no | both | RPC reply. Ephemeral topic, auto-cleans. |
| `strands/{peer}/camera/{cam}` | **NOT MQTT** — see §4.2 | – | – | metadata only | 128 KB MQTT payload limit. Camera frames go to S3; MQTT carries the URL. |
| `strands/{peer}/pose` | `strands/{peer}/pose` | 0 | no | up (filtered) | High-rate, route through Rules → Sitewise / TwinMaker. |
| `strands/{peer}/imu` | `strands/{peer}/imu` | 0 | no | up (filtered) | Same. |
| `strands/{peer}/odom` | `strands/{peer}/odom` | 0 | no | up (filtered) | Same. |
| `strands/{peer}/health` | `strands/{peer}/health` | 0 | yes | up | Slow (0.5 Hz). Retained = "show me last battery for offline robot". Rule → SNS on threshold breach. |
| `strands/{peer}/lidar/summary` | `strands/{peer}/lidar/summary` | 0 | no | up | Stats only. |
| `strands/{peer}/lidar/state` | `strands/{peer}/lidar/state` | 0 | yes | up | Like health. |
| `strands/{peer}/hand/{name}/state` | `strands/{peer}/hand/{name}/state` | 0 | no | **DOWN** (LAN-only) | 50 Hz, low-latency control input. Stays on Zenoh. Mirror only on operator demand. |
| `strands/{peer}/map/info` | `strands/{peer}/map/info` | 0 | yes | up | Slow, retained for joiners. |
| `strands/{peer}/safety/event` | `strands/{peer}/safety/event` | **2** | yes | up | At-most-once is unacceptable. QoS 2 (exactly once). Rule → DynamoDB (KMS-encrypted, point-in-time recovery) + SNS + Lambda. |
| `strands/safety/estop` | `strands/safety/estop` | **2** | yes | both | The single most important topic. See §6. |
| `strands/{peer}/input/{device}` | **NEVER bridges** | – | – | LAN-only | 50 Hz teleop. Internet RTT is fatal. |
| `strands/{peer}/stream` | `strands/{peer}/stream/meta` (gist) + Kinesis (full) | 0 | no | up (gist only) | VLA step at high rate. MQTT carries instruction + step + sizes; full payload to Kinesis Data Stream via Basic Ingest. |

### 4.2 Camera frames — why they don't ride MQTT

Today: `mesh/core.py` JPEG-encodes a frame, base64-wraps it, and publishes
`{shape, dtype, encoding, data}` on `strands/{peer}/camera/{cam}`. A 640×480
JPEG @ quality 80 is ~30–60 KB; base64 inflation puts it at 40–80 KB. Multiple
cams at 5–10 Hz easily clears 100 KB/s per robot.

AWS IoT MQTT max payload is **128 KB** per publish. Worse, the IoT message
cost model charges per-message *and* per-KB. Pushing camera frames through
MQTT is both technically dicey and economically nonsensical.

Right answer: **a thin camera offloader**.

```
robot                                      cloud
─────                                      ─────
Mesh._publish_cameras_once()
   ├─ if backend == bridge or iot:
   │     S3 PUT  s3://strands-frames/{peer}/{cam}/{ts_ns}.jpg
   │     IoT pub strands/{peer}/camera/{cam}/ref
   │             {peer_id, cam, t, shape, encoding, s3_uri,
   │              presigned_url, expires_at}
   └─ else (zenoh-only):
         existing inline-base64 path (LAN, fine)
```

Subscribers that want frames `GET` from S3 directly. The `ref` topic is
~300 bytes. Done.

### 4.3 Wildcards

Zenoh wildcards (`strands/*/presence`, `strands/{peer}/response/**`) map to
MQTT wildcards (`+` for one segment, `#` for tail). Our usage is exclusively
the patterns above, all of which translate verbatim.

---

<a id="5-shadow"></a>

## 5. State, identity, and the Device Shadow

### 5.1 The peer registry → Device Shadow mapping

Today's `PeerInfo` lives in process memory and is gone the moment the
discovering process exits. With AWS IoT, every robot has a **Thing**, and
each Thing has one or more **named shadows**. The shadow is authoritative
cloud-side state, JSON, versioned, with `desired` / `reported` / `delta`
semantics.

Mapping our presence payload to a `presence` named shadow:

```
$aws/things/{peer_id}/shadow/name/presence
{
  "state": {
    "reported": {
      "robot_type": "so100",
      "hostname": "g1-jetson-01",
      "connected": true,
      "hw": "so100_follower",
      "cameras": ["front", "wrist"],
      "inputs": [{"device": "leader", "method": "arm", "hz": 50.0}],
      "action_keys": ["shoulder_pan.pos", "shoulder_lift.pos", ...],
      "topics": ["pose", "imu", "health", "hand"],
      "task_status": "running",
      "instruction": "pick up the red block",
      "world": false
    }
  }
}
```

Our `Mesh._build_presence()` already constructs *exactly this dict*. The
bridge transport just needs to additionally publish it to the shadow's update
topic (`$aws/things/{peer_id}/shadow/name/presence/update`) instead of (or in
addition to) `strands/{peer_id}/presence`.

### 5.2 What this unlocks

- **Late joiners get full fleet state for free.** A Bedrock agent spinning
  up at 09:00 can read every robot's last-known capabilities via
  `IoT Data Plane GetThingShadow` without subscribing to anything. Today it
  has to wait one heartbeat per robot (≥0.5 s on a 2 Hz heartbeat).
- **Offline detection is cheap.** `connected: true` in shadow + `lastUpdated`
  more than 30 s old = offline. No registry-pruning thread needed
  cloud-side.
- **`desired` state becomes the assignment channel.** Want robot X to switch
  policy provider? Set `state.desired.policy_provider = "groot"`. Robot
  receives a `delta` MQTT message, applies, reports back.
- **Per-shadow auth.** IoT Policy can grant "write to my own presence shadow,
  read all `*/presence` shadows" cleanly. Today's `Mesh._on_presence` trusts
  any process on the LAN.

### 5.3 Identity — `peer_id` becomes Thing name

Today: `peer_id = f"{base}-{uuid.uuid4().hex[:8]}"`. Process-local. Anyone
can claim any peer_id.

Proposal: **the Thing name IS the peer_id**.

- Provisioned at first boot via Fleet Provisioning (claim cert → registration
  Lambda → real cert + Thing). The cert's CN matches the Thing name matches
  the peer_id.
- IoT Policy enforces it via `${iot:Connection.Thing.ThingName}` substitution
  (see §6).
- Bridge transport reads the Thing name from the cert and passes it as
  `peer_id=` to `init_mesh(...)`. Hard guarantee; no env override, no flag.
- Existing UUID scheme stays as the default for non-IoT deployments.
- For datasets / recordings already using UUID-based peer_ids, the registry
  carries an alias (`{display_name, iot_thing_name, legacy_uuid}`).

This is a **breaking change for any persisted dataset that hard-codes peer
ids in metadata**. Mitigation: a `peer_id_alias` field in the LeRobotDataset
metadata that the recorder writes as part of the bridge migration. Tests
already in `tests/test_dataset_recorder.py` need to grow a regression test
for the alias path.

---

<a id="6-safety"></a>

## 6. Auth, safety, and the E-Stop story

### 6.1 The current trust model

> Anyone who can `import zenoh` and reach the LAN multicast group can
> publish `strands/broadcast` with `{"action":"stop"}` and stop every robot
> on the network. They can also publish a fake `presence` claiming to be any
> peer_id.

This is fine for a lab. It is **not** fine for a fleet of customer-owned G1s
in twelve cities.

### 6.2 What AWS IoT gives us

X.509 mutual TLS, per-Thing certificates, IoT Policy with topic and action
ACLs, and CloudTrail audit on every operation.

Example IoT Policy for a robot Thing (publishing only its own data, listening
only on its own command inbox):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowOwnPresenceAndState",
      "Effect": "Allow",
      "Action": ["iot:Publish", "iot:Receive"],
      "Resource": [
        "arn:aws:iot:us-west-2:*:topic/strands/${iot:Connection.Thing.ThingName}/presence",
        "arn:aws:iot:us-west-2:*:topic/strands/${iot:Connection.Thing.ThingName}/state",
        "arn:aws:iot:us-west-2:*:topic/strands/${iot:Connection.Thing.ThingName}/pose",
        "arn:aws:iot:us-west-2:*:topic/strands/${iot:Connection.Thing.ThingName}/imu",
        "arn:aws:iot:us-west-2:*:topic/strands/${iot:Connection.Thing.ThingName}/odom",
        "arn:aws:iot:us-west-2:*:topic/strands/${iot:Connection.Thing.ThingName}/health",
        "arn:aws:iot:us-west-2:*:topic/strands/${iot:Connection.Thing.ThingName}/lidar/*",
        "arn:aws:iot:us-west-2:*:topic/strands/${iot:Connection.Thing.ThingName}/map/info",
        "arn:aws:iot:us-west-2:*:topic/strands/${iot:Connection.Thing.ThingName}/safety/event",
        "arn:aws:iot:us-west-2:*:topic/strands/${iot:Connection.Thing.ThingName}/response/*",
        "arn:aws:iot:us-west-2:*:topic/strands/${iot:Connection.Thing.ThingName}/stream/meta"
      ]
    },
    {
      "Sid": "AllowOwnCmdInbox",
      "Effect": "Allow",
      "Action": "iot:Subscribe",
      "Resource": [
        "arn:aws:iot:us-west-2:*:topicfilter/strands/${iot:Connection.Thing.ThingName}/cmd",
        "arn:aws:iot:us-west-2:*:topicfilter/strands/broadcast",
        "arn:aws:iot:us-west-2:*:topicfilter/strands/safety/estop"
      ]
    },
    {
      "Sid": "AllowOwnShadow",
      "Effect": "Allow",
      "Action": ["iot:Publish", "iot:Subscribe", "iot:Receive"],
      "Resource": [
        "arn:aws:iot:us-west-2:*:topic/$aws/things/${iot:Connection.Thing.ThingName}/shadow/*",
        "arn:aws:iot:us-west-2:*:topicfilter/$aws/things/${iot:Connection.Thing.ThingName}/shadow/*"
      ]
    }
  ]
}
```

Operator Things (Bedrock agents, fleet ops console) get a different policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowFleetCommand",
      "Effect": "Allow",
      "Action": "iot:Publish",
      "Resource": [
        "arn:aws:iot:us-west-2:*:topic/strands/*/cmd",
        "arn:aws:iot:us-west-2:*:topic/strands/broadcast"
      ]
    },
    {
      "Sid": "AllowFleetObserve",
      "Effect": "Allow",
      "Action": ["iot:Subscribe", "iot:Receive"],
      "Resource": [
        "arn:aws:iot:us-west-2:*:topicfilter/strands/+/presence",
        "arn:aws:iot:us-west-2:*:topicfilter/strands/+/state",
        "arn:aws:iot:us-west-2:*:topicfilter/strands/+/safety/event",
        "arn:aws:iot:us-west-2:*:topicfilter/strands/+/response/*"
      ]
    }
  ]
}
```

### 6.3 Emergency stop, defence in depth

`Mesh.emergency_stop()` today does two things:

1. `broadcast({"action": "stop"}, timeout=3.0)` — peers actually stop.
2. Audit and record.

In bridge mode, E-stop becomes **three independent paths**, any of which is
sufficient:

1. **Local Zenoh broadcast** (LAN-fast; the current behaviour) — peers in
   the same broadcast domain stop in <10 ms.
2. **MQTT broadcast on `strands/broadcast`** — every robot worldwide subscribed
   to `strands/broadcast` receives the stop. Latency = MQTT one-way, ~50 ms
   typical.
3. **IoT Rule → Lambda fan-out** — the rule matches `WHERE action = 'stop'`
   on `strands/broadcast` and additionally invokes a Lambda that iterates
   the fleet and publishes individual `strands/{peer}/cmd` `{action:"stop"}`
   messages. This catches robots that for whatever reason missed the
   `strands/broadcast` subscription.

The *audit* path also gets defence in depth:

- Local JSONL stays the source of truth (immutable, on-host, unaffected by
  cloud outages).
- IoT Rule mirrors safety events to a DynamoDB table (`mesh-safety-events`)
  with KMS encryption and point-in-time recovery enabled.
- `audit.py` grows a merkle hash chain (each line includes
  `prev_sha256`) so the cloud mirror can verify nothing was deleted from the
  local log retroactively.

### 6.4 Audit log split-brain

Question: when the local JSONL says one thing and DynamoDB says another, who
wins?

Answer: **local wins**. The local log is the legal record. Cloud is a
queryable mirror. The hash chain lets us prove the cloud copy is consistent
with what was actually written on-host. If the cloud copy diverges, the cloud
mirror is broken; the robot's record is canonical. This matters for incident
response — a compromised cloud account cannot rewrite a safety history that
already hit local disk.

---

<a id="7-costs"></a>

## 7. The honest costs, latencies, and footguns

### 7.1 Latency budget

| Path | Typical | Worst | Acceptable for |
|---|---|---|---|
| Zenoh, same process | <0.1 ms | 1 ms | Anything |
| Zenoh, same LAN | <1 ms | 5 ms | Anything including teleop and visual servo |
| AWS IoT Core, same region | 30–80 ms | 200 ms | Presence, state, RPC, safety, fleet ops |
| AWS IoT Core, cross-region | 80–200 ms | 500 ms | Fleet ops, audit |
| AWS IoT Core, mobile uplink | 100–500 ms | 2000+ ms | Status reporting only |

**Hard rule:** no closed-loop control crosses MQTT. Teleop, hand control,
visual servo, force feedback — Zenoh-only.

### 7.2 Cost model

IoT Core pricing (us-west-2, mid-2026 figures):

- Connections: $0.08 per million minutes connected.
- Messaging: $1.00 per million messages (≤5 KB/message).
- Rules engine: $0.15 per million rules triggered.
- Device Shadow: $1.25 per million operations.
- Basic Ingest (rules-only routing): **free** for messages destined to other
  AWS services (Kinesis, Firehose, Lambda, IoT Analytics, Sitewise,
  Timestream, S3 via Firehose, SNS, SQS, DynamoDB).

Naive rollout: 100 robots × 10 Hz state + 2 Hz presence + 0.5 Hz health + …

```
state:    100 robots × 10 Hz × 86400 s = 86,400,000 msg/day
presence: 100 × 2 × 86400              = 17,280,000 msg/day
health:   100 × 0.5 × 86400            =  4,320,000 msg/day
pose:     100 × 10 × 86400             = 86,400,000 msg/day
imu:      100 × 10 × 86400             = 86,400,000 msg/day
                                        ───────────────────
                                         280,800,000 msg/day
```

At $1/M = $281/day = **$8,400/month** for 100 robots. Painful.

Mitigations (these matter and they're not optional):

1. **Use Basic Ingest for telemetry.** State/pose/imu/odom/health go to
   Kinesis or Timestream via Rules. Basic Ingest charges $0 for the
   ingest leg. Subscribers (operator dashboard) read from Timestream
   instead of subscribing to MQTT live. Cost drops to ~$0 for telemetry,
   we pay per-Kinesis-shard instead (predictable).
2. **Filter high-rate topics from the bridge by default.** `STRANDS_MESH_BRIDGE_TOPICS=presence,health,safety/event,cmd,response,broadcast` is the safe default. Pose / IMU / state opt-in.
3. **Aggregate before publishing.** A robot can buffer state at 10 Hz locally and publish a batched envelope at 1 Hz. Saves 90% of messages with no operational loss for fleet ops.
4. **Don't subscribe state from MQTT.** Read from Timestream / Sitewise /
   S3+Athena. MQTT subscription costs are not free.

### 7.3 Payload size

128 KB MQTT max. Camera frames don't fit (see §4.2). VLA `stream` payloads
with full observation tensors might not fit either. Solution: publish a
"meta" topic with sizes + S3 references; full payload to Kinesis Data Stream
via Basic Ingest (1 MB record limit).

### 7.4 Offline behaviour

Robot loses internet → MQTT publishes fail. Without intervention:

- Local Zenoh keeps working — LAN ops continue.
- State / presence / safety events queue up… nowhere.
- Reconnect → no replay. Cloud has a gap.

Two choices:

(a) **Greengrass v2 on the robot.** Local broker, local Lambda components,
    store-and-forward queue. Reconnect replays. Cost: ~200 MB Java runtime,
    not viable on a Jetson Nano. Fine on a G1's Orin or a desktop ops box.

(b) **Lighter custom store-and-forward.** A small SQLite-backed queue inside
    `IotMqttTransport`: failed publishes go to a ring buffer; reconnect drains
    it oldest-first with a configurable max-replay-rate so you don't DoS
    yourself on reconnect. ~500 lines of Python. **My preferred default.**

### 7.5 Rule-engine selectivity

VLA `stream` at 50 Hz × N robots will overwhelm anything subscribing live.
Rules MUST filter aggressively. Examples:

- `WHERE instruction != ''` — only forward instructed steps.
- `WHERE step % 10 = 0` — downsample to 5 Hz.
- `WHERE robot_type IN ('g1', 'gr1')` — humanoids only.

### 7.6 Region / availability

IoT Core is regional. A robot in Tokyo connecting to `us-west-2` adds 100+ ms
RTT. For multi-region fleets we want one IoT broker per region with a
cross-region replication Lambda that mirrors `presence` + `safety/event` to a
"hub" region for global dashboards. Don't replicate `state` or `pose` —
volume kills the cost model.

---

<a id="8-architecture"></a>

## 8. Proposed architecture (still no implementation)

```
┌─────────────────────────────── Robot Host ───────────────────────────────┐
│                                                                          │
│  Robot()/Simulation()  ──►  Mesh (UNCHANGED)                             │
│                                │                                         │
│                                │ put() / declare_subscriber()            │
│                                ▼                                         │
│                          MeshTransport (NEW abstract layer)              │
│                                │                                         │
│                ┌───────────────┼─────────────────────┐                   │
│                ▼               ▼                     ▼                   │
│         ZenohTransport   IotMqttTransport     BridgeTransport            │
│                │               │                     │                   │
│                │               │           ┌─────────┴────────┐          │
│                │               │           ▼                  ▼          │
│                │               │    ZenohTransport    IotMqttTransport   │
│                │               │           │                  │          │
│                │               │           │       (filter: presence,    │
│                │               │           │        health, safety/event,│
│                │               │           │        cmd, response, etc.) │
│                │               │           │                  │          │
│                ▼               ▼           ▼                  ▼          │
│           LAN peers        AWS IoT       LAN peers       AWS IoT Core    │
│         (multicast)         Core      (multicast)         (mTLS)         │
│                                                                          │
│  audit.py (UNCHANGED) ── local JSONL with merkle hash chain              │
│         │                                                                │
│         └─ also publishes to strands/{peer}/safety/event                 │
│            ─► MQTT ─► Rule ─► DynamoDB (KMS, PITR) + SNS + Lambda        │
└──────────────────────────────────────────────────────────────────────────┘

                            AWS IoT Core (mTLS, MQTT v5)
                                       │
       ┌───────────────────────────────┼───────────────────────────────┐
       │                               │                               │
       ▼                               ▼                               ▼
   Device Shadows               Rules Engine                 Cross-Region
   (presence, state              SQL filter +                Replication
   per Thing)                    routing                     Lambda
       │                          │                              │
       │                  ┌───────┼──────────┬─────────┐         │
       │                  ▼       ▼          ▼         ▼         │
       │             Kinesis    Lambda    DynamoDB  SNS/SQS      │
       │             Firehose  (E-stop,  (safety,  (alerts)      │
       │                ▼      alerts,   audit)                  │
       │              S3 +     fleet                             │
       │             Athena   provisioning)                      │
       │                                                         │
       └─►  Bedrock Agent / Fleet Ops Console / TwinMaker  ◄──────┘
              │
              └─ uses GetThingShadow, IoT Data Plane Publish,
                 invokes robot_mesh tool actions
```

The crucial thing: `Mesh`, `SensorLoopsMixin`, `audit.py`, `tools/robot_mesh.py`,
`Robot()`, `Simulation()`, `HardwareRobot` — **none of them change**. They
all already talk through `put()` / `declare_subscriber()`.

---

<a id="9-mesh-transport"></a>

## 9. `MeshTransport` protocol — the one refactor

This is the only invasive change to PR #101's code, and it's small.

### 9.1 The protocol

```python
# strands_robots/mesh/transport/base.py (NEW)

from typing import Protocol, Callable, Any

class SubHandle(Protocol):
    """Opaque subscription handle. Must support .undeclare()."""
    def undeclare(self) -> None: ...


class MeshTransport(Protocol):
    """Pluggable transport for Mesh. Replaces direct zenoh.Session usage.

    Implementations:
        - ZenohTransport      — current behaviour, LAN multicast
        - IotMqttTransport    — AWS IoT Core, mTLS, single-broker
        - BridgeTransport     — both, with a topic filter for what bridges up

    Lifetime: ref-counted singleton per process, just like the current
    get_session()/release_session() pair. Construction is lazy and
    backend-driven by STRANDS_MESH_BACKEND.
    """

    def put(self, key: str, data: dict[str, Any]) -> None:
        """Publish a JSON payload. Fire-and-forget. No-op on failure."""
        ...

    def declare_subscriber(
        self, key_expr: str, handler: Callable[[Any], None]
    ) -> SubHandle:
        """Subscribe to a key expression. Handler receives a Sample-like obj
        with .key_expr and .payload.to_bytes(). Backend must adapt MQTT
        messages to that shape."""
        ...

    def is_alive(self) -> bool: ...

    def close(self) -> None:
        """Tear down. Idempotent. Must release any underlying connection."""
        ...
```

### 9.2 The diff against `session.py`

**Change in `mesh/session.py`** (sketch — no code lands today):

```python
# Before
def put(key, data):
    if _SESSION is None: return
    _SESSION.put(key, json.dumps(data).encode())

# After
def put(key, data):
    transport = _current_transport()
    if transport is None or not transport.is_alive(): return
    transport.put(key, data)
```

`get_session()` / `release_session()` get renamed to
`get_transport()` / `release_transport()` (or stay as compat shims that
call into the new functions). The rest of `session.py` is unchanged.

### 9.3 The Sample shim

Zenoh callbacks receive a `Sample` object with `.key_expr` and
`.payload.to_bytes()`. MQTT callbacks receive a topic string and a `bytes`
payload. The transport adapter wraps MQTT callbacks in a tiny shim:

```python
class _MqttSampleShim:
    def __init__(self, topic: str, payload: bytes):
        self.key_expr = topic
        self.payload = _PayloadShim(payload)

class _PayloadShim:
    def __init__(self, b: bytes): self._b = b
    def to_bytes(self) -> bytes: return self._b
```

Now every existing handler in `mesh/core.py` (`_on_presence`, `_on_cmd`,
`_on_response`, the user `subscribe()` handler) works against MQTT *without
modification*.

### 9.4 What this changes outside mesh/

Nothing. The `Mesh` class keeps calling `session.declare_subscriber(...)`. The
`tools/robot_mesh.py` keeps calling `mesh.send()`. `Robot()` keeps doing
`init_mesh(robot, peer_id, peer_type, mesh)`. All tests in `tests/test_mesh*`
keep working because they exercise `Mesh` semantics, not Zenoh APIs.

---

<a id="10-migration"></a>

## 10. Migration plan — 4 layers, each independently shippable

Each layer is a standalone PR. Each one is independently useful, even if the
next one never lands.

### Layer 1 — `MeshTransport` extraction (pure refactor, no behaviour change)

**Scope**: introduce `mesh/transport/base.py` (the protocol), `mesh/transport/zenoh.py`
(the current behaviour, lifted as-is from `session.py`), and adapt
`mesh/session.py` to delegate. Add a `STRANDS_MESH_BACKEND=zenoh` env var
that's the default and the only valid value at this layer.

**Risk**: low. ~300 lines moved, zero new dependencies, zero functional
change. Existing tests must continue to pass *unchanged*.

**Deliverable**: PR titled `refactor(mesh): extract MeshTransport protocol,
keep zenoh as default`.

### Layer 2 — `IotMqttTransport`

**Scope**: implement `mesh/transport/iot.py` using `awscrt.mqtt5` (the AWS IoT
Device SDK v2 transport). Cert/key load from `~/.strands_robots/iot/{cert,key,ca}`
or env vars. Topic schema = identity (no translation needed). Wildcard
mapping table. SQLite-backed store-and-forward queue. Reconnect with
exponential backoff. Configurable QoS per-topic via a small config table.

**Optional dep**: `[mesh-iot]` extra → `awscrt>=0.20.0`,
`awsiotsdk>=1.21.0`. Stays out of `[mesh]` so existing users aren't
forced to install AWS dependencies.

**New env**: `STRANDS_MESH_BACKEND=iot`,
`STRANDS_IOT_ENDPOINT=xxxxxxxxxxxxxx-ats.iot.us-west-2.amazonaws.com`,
`STRANDS_IOT_THING_NAME` (overrides peer_id), `STRANDS_IOT_CERT_DIR`.

**Tests**: `tests_integ/test_iot_transport.py` with a moto / localstack
broker, plus a real IoT integration test gated on `pytest.importorskip("awscrt")`
and AWS creds.

**Deliverable**: PR titled `feat(mesh): IotMqttTransport — AWS IoT Core
backend`.

### Layer 3 — `BridgeTransport` + Device Shadow + camera S3 offload

**Scope**: implement `mesh/transport/bridge.py` that owns both a
`ZenohTransport` and an `IotMqttTransport` and forwards based on a topic
filter. Add `mesh/cloud/shadow.py` that mirrors `presence` payloads to the
named shadow on every heartbeat. Add a `mesh/cloud/camera_offload.py` that
short-circuits `_publish_cameras_once()` to S3 + reference topic when the
backend is `bridge` or `iot`.

**New env**: `STRANDS_MESH_BACKEND=bridge`,
`STRANDS_MESH_BRIDGE_TOPICS=presence,health,safety/event,safety/estop,cmd,response,broadcast`
(default — explicit allow-list, denies pose/imu/odom/state/input by default),
`STRANDS_MESH_CAMERA_S3_BUCKET`.

**Tests**: full round-trip test with one sim peer on Zenoh and one Lambda
listener on IoT, exchanging RPCs.

**Deliverable**: PR titled `feat(mesh): BridgeTransport + Device Shadow +
camera S3 offload`.

### Layer 4 — Operations: Rules / Lambda / Dashboards / Provisioning

**Scope**: ship a CDK/Terraform package under `infrastructure/` that
provisions:

- IoT Rules: state→Kinesis→Timestream, safety→DynamoDB+SNS, presence→shadow
  reconciler Lambda, estop→fan-out Lambda.
- DynamoDB table `mesh-safety-events` with KMS + PITR.
- IoT Policies: robot policy template + operator policy template.
- Fleet Provisioning template + registration Lambda (claim cert → real cert).
- IoT Jobs templates for policy/data_config rollouts.
- A reference Bedrock-agent integration that uses the `robot_mesh` tool
  with the bridge backend.

**Optional**: a Grafana / QuickSight dashboard JSON for the Sitewise asset
model.

**Tests**: deploy to a dev sub-account, smoke-test against a live G1 sim.

**Deliverable**: PR titled `feat(mesh): operations infra — IoT Rules, Jobs,
Fleet Provisioning, dashboards`.

---

<a id="11-demos"></a>

## 11. Killer demos this enables

### 11.1 Cross-continent E-stop

```
Operator (Seattle laptop, Bedrock agent) ──► robot_mesh.emergency_stop()
                                              │
                                              ▼
                                       MQTT broadcast on
                                       strands/safety/estop
                                              │
                       ┌──────────────────────┼──────────────────────┐
                       ▼                      ▼                      ▼
                 G1 in Tokyo            SO-101 in Berlin     ALOHA in Austin
                 stops in <100 ms        stops in <100 ms     stops in <100 ms
```

Every stop is logged in CloudTrail (who issued it), DynamoDB (the safety event
itself), local JSONL (immutable host record), and the merkle hash chain
verifies cloud and local agree.

### 11.2 Bedrock agent driving a real robot through a browser

`mesh.html` (browser) → WebSocket → DevDuck → `robot_mesh(action="tell",
target="g1-prod-04", instruction="set the table")`. Tool call goes through the
bridge transport → IoT Core → Greengrass on the G1 → local Mesh → robot
executes. Round trip with E-stop available at every step. mTLS authenticated,
IAM-audited.

### 11.3 Cross-region swarm

50 SO-101s in `ap-northeast-1` and 50 in `us-west-2`. A `broadcast` published
in `us-west-2` is replicated to `ap-northeast-1` via the cross-region Lambda
(see §7.6). Both fleets execute. Responses come back via per-region
`response/{turn_id}` topics, aggregated by the operator's local Mesh.

### 11.4 Greengrass-in-a-G1

A G1 humanoid runs the local Zenoh router AND a Greengrass core. Customer's
home internet drops. The G1 keeps working locally — leader arm in the same
room still teleops the follower because Zenoh is unaffected. Cloud queue
fills with state / presence / safety. Internet returns, Greengrass replays
the queue oldest-first, the cloud shadow gets a delta update, audit log gets
the missing safety events.

### 11.5 TwinMaker mirror

Every `Simulation` instance pushes `world.objects` to a TwinMaker scene. The
ops team's Grafana panel shows real robots + digital twins side-by-side.
Click a real robot → see its last 1000 frames of state at 10 Hz pulled from
Timestream. Click a sim → see the same fields, same panel, no schema
divergence.

### 11.6 Sitewise + QuickSight

200 robots × `health` topic → IoT Rule → Sitewise asset model with
`battery_pct`, `cpu_load`, `disk_free_gb`, `mem_pct`, `uptime_s`, `temps.*`
as measurements. QuickSight dashboard with a 5-click setup. Battery alerts
via SNS triggered by Sitewise alarms.

---

<a id="12-risks"></a>

## 12. Open questions / unresolved risks

These are the things I want pushback on before any code lands.

1. **Who owns peer_id?** Today it's process-local. With IoT it must equal
   the Thing name. That's a breaking change for any persisted dataset that
   records peer_ids. Mitigation: dataset metadata grows
   `peer_id_alias = {iot_thing_name, legacy_uuid, display_name}`; the
   `DatasetRecorder` writes all three. Question: is the Thing-name
   constraint okay, or do we want a registry indirection so peer_id can be
   anything but maps to a Thing?

2. **RPC over MQTT v3 vs v5.** v5 has `correlationData` which is a strictly
   better fit for our `turn_id` scheme. AWS IoT Core supports both, but
   pricing differs slightly and some tooling is v3-biased. Default to v5;
   keep v3 fallback.

3. **Camera frame round-trip.** S3 PUT + presigned URL adds 100–500 ms vs
   inline-base64 LAN Zenoh. Probably fine for monitoring, fatal for
   closed-loop visual servoing. The bridge transport must keep cameras on
   Zenoh and only mirror `camera/*/ref` (metadata) to MQTT. Verified: this
   matches the §4.2 design.

4. **Fleet provisioning at scale.** Without it, every robot needs a
   hand-issued cert. With it, you need a claim-cert / Lambda registration
   flow — non-trivial but standard. First robot → manual cert. Fleet >
   10 → use FP. Where's the cutoff for shipping the FP template?

5. **Greengrass weight.** ~200 MB Java install. Fine on Orin / desktop. Not
   landing on Jetson Nano. Lighter alternative: SQLite store-and-forward
   inside `IotMqttTransport`. My preferred default. Greengrass is opt-in
   for sites that need component lifecycle / local Lambdas / HSM creds.

6. **Audit log split-brain.** Local JSONL and DynamoDB can disagree.
   Resolution: local wins (it's the legal record). Cloud is a queryable
   mirror. Merkle hash chain in `audit.py` lets cloud verify nothing was
   deleted from local. Question: do we also want a tamper-evident write to
   AWS CloudTrail Lake or a separate IoT-side audit S3 bucket with object
   lock?

7. **Cost ceiling.** $8.4k/month for 100 robots at naive ingest is real.
   Basic Ingest mitigates. We must ship the bridge with Basic Ingest as the
   default for telemetry from day one or someone will get a surprise bill.

8. **Topic schema versioning.** If we ever need to evolve the JSON shape of
   `state` or `presence`, MQTT subscribers in the wild will break. We need
   a `v` field on every payload from the start. Today we have `t` but no
   `v`. Add `v: 1` to all payloads as a no-op preparatory change in the
   Layer 1 PR.

9. **Multi-tenant fleets.** If two customers' robots end up in the same IoT
   account (e.g. a managed-service offering), topic isolation needs work.
   Either prefix every topic with `{customer_id}/strands/...` or use
   separate Things-Group-scoped policies. Defer until we actually have
   multi-tenant.

10. **Reconnect storms.** A regional outage that flaps will cause every
    robot to reconnect simultaneously. SDK has built-in jitter; we should
    set explicit `min_reconnect_timeout_secs=5, max_reconnect_timeout_secs=60`
    rather than defaults, and add a randomized startup delay of up to
    30 s on the first connect so a fleet-wide reboot doesn't thunderhead.

---

<a id="13-non-goals"></a>

## 13. Decision log — what we explicitly chose NOT to do

These are real options that we considered and decided against, with the
reasons recorded so future-us doesn't redo the analysis.

- **Replace Zenoh with MQTT entirely.** Rejected. Zenoh's <1 ms LAN
  performance and multicast scouting are non-negotiable for teleop and
  visual servo. MQTT is for WAN.

- **Use IoT Core as the only transport, with regional deployments doing the
  LAN role.** Rejected. Even same-AZ RTT is 1–5 ms minimum; you'd never
  approach Zenoh in-process latency.

- **Bridge every topic by default.** Rejected. Cost and bandwidth blow up.
  Default bridges are an explicit allow-list.

- **Use protobuf instead of JSON.** Rejected at this layer. JSON is what
  PR #101 chose and it works fine on MQTT. Migration to protobuf is a
  separate decision orthogonal to this one and probably should wait until
  there's a measured need (e.g. cross-language fleet operators).

- **Run our own MQTT broker (Mosquitto, EMQX) on EC2.** Rejected. Operating
  a multi-region, mTLS-terminating, audited MQTT broker is a non-trivial
  ops burden. AWS IoT Core does it managed for less than the cost of the
  EC2s plus the ops time.

- **Use ROS 2 / DDS.** Rejected for the bridge layer. ROS 2 is a fine LAN
  transport (and overlaps Zenoh's territory — ros2-zenoh-bridge exists for
  that exact reason), but the cloud-edge story isn't there. We'd end up
  bridging to MQTT anyway.

- **Use AWS Kinesis Video Streams for camera frames.** Rejected as the
  default. KVS has a real cost and a substantial integration surface; S3 +
  presigned URL is enough for "monitoring" cameras. KVS becomes a future
  extra for fleets that need actual real-time video search / replay.

- **Store the audit log only in DynamoDB.** Rejected. Local JSONL is the
  legal record; the cloud is a mirror. A compromised cloud account must
  not be able to rewrite history that already hit local disk.

---

<a id="14-next-steps"></a>

## 14. Next steps

1. **Land PR #101** (`autonomous/mesh-session`) as-is, no changes. It's
   correct and good.
2. **Open a Layer-1 PR** that introduces `mesh/transport/base.py` and
   `mesh/transport/zenoh.py`, refactors `mesh/session.py` to delegate, and
   adds `v: 1` to every payload. Zero behaviour change. ~300 lines.
   Includes a regression test that asserts the wire format hasn't changed.
3. **Spike `IotMqttTransport`** in a feature branch (`feat/iot-transport`)
   against a sandbox AWS account. Goal: prove `Mesh.send()` works
   end-to-end against IoT Core unchanged. ~1 week. Does NOT merge yet —
   informs the Layer 2 PR.
4. **Write the IoT Policy templates** (robot + operator) and validate them
   in a dev account against a real AWS Bedrock agent invoking
   `robot_mesh(action="tell", target=...)`. This validates §6 in
   practice before any code commits.
5. **Decide on Greengrass vs SQLite store-and-forward** based on the spike.
   My current recommendation is SQLite default + Greengrass opt-in.
6. **Land Layer 2** as `feat(mesh): IotMqttTransport`.
7. **Land Layer 3** as `feat(mesh): BridgeTransport + Device Shadow +
   camera S3 offload`.
8. **Land Layer 4** as `feat(mesh): operations infra`.
9. **Update README.md** with one paragraph and the architecture diagram.

The README paragraph I'd write the day Layer 3 lands:

> strands-robots ships a peer-to-peer mesh that runs on Eclipse Zenoh for
> the LAN hot loop (sub-millisecond, multicast, peer-to-peer) and bridges
> to AWS IoT Core for global fleet ops, durable shadow state, and
> policy-gated remote control. Same topic scheme, same `robot_mesh` tool,
> two transports.

---

## Appendix A — Full topic-to-MQTT translation table

(Authoritative reference. If something is published in `mesh/core.py` or
`mesh/sensors.py` and not on this list, it's a bug in the design — open
an issue.)

| Source (Zenoh key) | MQTT topic | QoS | Retain | Direction | Bridge by default? | Notes |
|---|---|---|---|---|---|---|
| `strands/{peer}/presence` | identical | 1 | yes | up + down | yes | Also writes `$aws/things/{peer}/shadow/name/presence/update` |
| `strands/*/presence` | `strands/+/presence` | 1 | – | sub | yes | Wildcard maps `*` → `+` |
| `strands/{peer}/state` | identical | 0 | no | up | **no** (opt-in) | High volume; route via Basic Ingest → Timestream |
| `strands/{peer}/cmd` | identical | 1 | no | up + down | yes | RPC inbox; bridge needed for remote ops |
| `strands/broadcast` | identical | 1 | no | up + down | yes | Fan-out RPC |
| `strands/{peer}/response/{turn}` | identical | 1 | no | up + down | yes | Single-segment turn id; ephemeral topic |
| `strands/{peer}/response/**` | `strands/{peer}/response/#` | 1 | – | sub | yes | Tail wildcard |
| `strands/{peer}/camera/{cam}` | not bridged | – | – | LAN-only | no | Replaced by `/ref` topic + S3 |
| `strands/{peer}/camera/{cam}/ref` | identical | 0 | no | up | yes | New topic: S3 pointer + metadata |
| `strands/{peer}/pose` | identical | 0 | no | up | no (opt-in) | Volume; Basic Ingest |
| `strands/{peer}/imu` | identical | 0 | no | up | no (opt-in) | Volume; Basic Ingest |
| `strands/{peer}/odom` | identical | 0 | no | up | no (opt-in) | Volume; Basic Ingest |
| `strands/{peer}/health` | identical | 0 | yes | up | yes | Slow; retained for offline visibility |
| `strands/{peer}/lidar/summary` | identical | 0 | no | up | no (opt-in) | Volume |
| `strands/{peer}/lidar/state` | identical | 0 | yes | up | yes | Slow; retained |
| `strands/{peer}/hand/{name}/state` | not bridged | – | – | LAN-only | no | 50 Hz control input |
| `strands/{peer}/map/info` | identical | 0 | yes | up | yes | Slow; retained |
| `strands/{peer}/safety/event` | identical | **2** | yes | up | yes | Exactly-once + retained for late subscribers |
| `strands/safety/estop` | identical | **2** | yes | up + down | yes | Defence-in-depth: Lambda fan-out also subscribes |
| `strands/{peer}/input/{device}` | not bridged | – | – | LAN-only | no | 50 Hz teleop; LAN only |
| `strands/{peer}/stream` | not bridged | – | – | LAN-only | no | High-rate VLA stream |
| `strands/{peer}/stream/meta` | identical | 0 | no | up | yes | New topic: instruction + step + sizes + S3/Kinesis ref |

## Appendix B — Glossary

- **Mesh** — `strands_robots.mesh.core.Mesh`. The peer-to-peer component
  that every Robot and Simulation owns.
- **Peer** — anything publishing on `strands/*/presence`. A robot, sim,
  agent, fleet ops console, anything.
- **peer_id** — the unique identifier for a peer. Today: random suffix.
  Proposed under IoT: AWS IoT Thing name.
- **Transport** — the wire-level pub/sub mechanism. Today: Zenoh.
  Proposed: pluggable, with Zenoh + IoT MQTT + Bridge implementations.
- **Bridge transport** — both Zenoh and IoT, with a topic filter for
  what crosses the WAN.
- **Basic Ingest** — AWS IoT feature where messages destined for AWS
  services via the rules engine are not billed at the per-message MQTT
  rate. Critical for our cost model.
- **Named shadow** — AWS IoT Device Shadow, scoped by name. We use one
  named shadow per topic family that benefits from durable state
  (`presence`, eventually `health`, `task`).
- **Greengrass v2** — AWS IoT's edge runtime. Local MQTT broker, local
  Lambda components, store-and-forward, HSM creds.
- **Fleet Provisioning** — AWS IoT's mechanism for claiming a real
  device cert from a bootstrap claim cert, on first boot, via a
  registration Lambda.

