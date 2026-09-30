# Bridges

At the end of this page your fleet's presence, commands and safety events reach AWS IoT Core over MQTT5 with per-robot X.509 identities while joint state and camera frames stay on the LAN, and you can write the Zenoh ACL separating operators from robots.

```bash
pip install 'strands-robots[mesh-iot]'          # awsiotsdk, awscrt, boto3 on top of [mesh]
```

```python title="sketch"
from strands_robots.mesh.iot import bootstrap_account, provision_robot

bootstrap_account()                                   # once per account and region: Rules, Lambda, DynamoDB audit, Fleet Provisioning template
p = provision_robot("so101-arm-01")                   # per robot: cert + policy + Thing named after the peer
print(p.env_vars())                                   # STRANDS_IOT_THING_NAME, STRANDS_IOT_ENDPOINT, STRANDS_IOT_CERT_DIR, STRANDS_MESH_BACKEND=iot
```

Export them, set `STRANDS_MESH_BACKEND=bridge`, and `Robot("so101", mode="real", mesh=True, peer_id="so101-arm-01")` publishes on both wires.

## Three backends

| `STRANDS_MESH_BACKEND` | transport | for |
|---|---|---|
| `zenoh` (default) | `ZenohTransport`: one LAN session per process, mTLS and ACL from `STRANDS_MESH_*` | a room, a lab |
| `iot` | `IotMqttTransport`: MQTT5 over mTLS to AWS IoT Core, no Zenoh | a robot whose peers are all in the cloud |
| `bridge` | `BridgeTransport`: one of each; every `put` fans out, subscriptions fan in | production: LAN peers plus dashboards, audit and fleet ops in AWS |

The bridge degrades rather than fails: Zenoh down means pure IoT, IoT down pure Zenoh, both down `is_alive()` false, every put a no-op.

## What the bridge forwards

The MQTT side is filtered by topic suffix. On both wires by default: `presence`, `health`, `cmd`, `response`, `broadcast`, `safety/event`, `safety/estop`, `safety/resume`. LAN-only: `state`, `pose`, `imu`, `odom`, `camera`, `input`, `hand`, `stream`. `STRANDS_MESH_BRIDGE_TOPICS`, a comma-separated suffix list, replaces the default. Inbound duplicates (a presence heard on both wires) are dropped at the `Mesh` layer by `sender_id` and `turn_id`; `STRANDS_MESH_BRIDGE_DEDUP_STRICT` tightens that.

Keys are unchanged on MQTT (`strands/<peer>/cmd` is a valid topic); wildcards map `*` to `+`, `**` to `#`.

## The IoT trust model

Each robot is a Thing whose name equals its mesh `peer_id` and cert CN; the MQTT `client_id` is set to it so `${iot:Connection.Thing.ThingName}` in the IoT policy scopes every robot to its own topics. `provision_robot(thing_name, region=, cert_dir=, attributes=, allow_estop_publish=False)` generates the key locally, has AWS sign a CSR (`CN=<thing>, O=strands-robots`) and writes `<thing>.cert.pem`, `<thing>.private.key` and `AmazonRootCA1.pem` under `STRANDS_IOT_CERT_DIR` (default `~/.strands_robots/iot`). Re-running publishes a changed policy document as the default. A second policy, `strands-robot-children`, covers the Thing's child peers: a simulation attaches each robot as `<thing>__<robot>`, published over the Thing's session on `strands/<thing>__*/*`. AWS IoT ends a session on an ungranted publish, so a certificate without it reconnects every child heartbeat; the transport WARNs with the topic and `strands-robots iot reprovision <thing>` attaches it; Thing names cannot contain `__`.


| variable | meaning |
|---|---|
| `STRANDS_IOT_ENDPOINT` | the account's ATS endpoint |
| `STRANDS_IOT_THING_NAME` | the Thing, equal to cert CN and peer id |
| `STRANDS_IOT_CERT_DIR` | where the three PEM files live |
| `STRANDS_IOT_CA_FILE` | overrides the root CA path |
| `STRANDS_IOT_DIRECT_AUTH` | `x509` (default with a cert) or `sigv4` (IAM) for direct messages |

Missing `awsiotsdk`, endpoint or cert files make `connect()` return `False` with an ERROR line; the mesh stays off. A reconnect is a clean session; the transport re-subscribes every topic filter and WARNs with the list.

Cloud mirrors: `shadow.enable_for_mesh(mesh)` keeps a Device Shadow (`presence`) per Thing; `camera_offload.enable_for_mesh(mesh)` puts frames in S3 (`STRANDS_MESH_CAMERA_S3_BUCKET`, `STRANDS_MESH_CAMERA_S3_PREFIX`), publishing presigned URLs (`STRANDS_MESH_CAMERA_PRESIGN_TTL`).

## The Zenoh ACL

Under `STRANDS_MESH_AUTH_MODE=mtls` the built-in ACL lets any CA-signed peer publish and subscribe anywhere, and `Mesh.start` refuses that posture until `STRANDS_MESH_ACCEPT_PERMISSIVE_ACL=1` is set or `STRANDS_MESH_ACL_FILE` names a JSON5 file. The template `examples/mesh/mesh_acl_example.json5` has `default_permission: "deny"`, named `rules`, `subjects` keyed by literal `cert_common_names`, and `policies` binding the two. Its three roles:

| subject | may |
|---|---|
| `robot_peer` | publish telemetry; subscribe to presence, broadcast, safety, its own `cmd` and `response` |
| `operator_peer` | publish commands; subscribe `**` |
| `dashboard_peer` | subscribe to observe; publish presence |

Zenoh 1.x facts the loader verified live: `enabled: true` is required, else the block is a no-op; `cert_common_names` match literally (no globs), so enumerate every CN; omitting `interfaces` matches every link, `[]` is rejected; `key_exprs` see the key without the namespace: `**/cmd` works, `strands/*/cmd` matches nothing; `declare_subscriber` rules live in the `egress` flow, `put` rules in `ingress`. No `${cn}` interpolation exists, so strict per-peer isolation on Zenoh means one rule per robot CN (`mesh_acl_strict_per_peer.json5`); IoT isolates by construction. Two file shapes: `default_permission: "deny"` plus `allow` rules is a whitelist, a gap denies; `default_permission: "allow"` plus rules is a blacklist, an unnamed key expression is open on the wire. `_parse_acl_bytes` (`mesh/_acl_config.py`) refuses the blacklist shape with `PermissiveACLError` unless `STRANDS_MESH_ACCEPT_PERMISSIVE_ACL` is `1`, `true` or `yes`. Two more readers: `Mesh._refuse_under_permissive_default_acl` (`mesh/core.py`), the start gate under the built-in permissive default, and `session._build_config` (`mesh/session.py`), whose no-ACL-file WARNING is suppressed by it; `check_mesh` (`doctor.py`) reports all three. A token set to load a blacklist in CI so waives the start gate too: drop the file later and the fleet runs wire-open with no log signal. Never set it on a production fleet.

`STRANDS_MESH_CA_PINS` pins the CA fingerprints; `STRANDS_MESH_DISABLE_CA_PIN` turns pinning off in a lab.

## ROS 2 as a peer

`RosBridgedRobot`, `RosbridgeRobot` and `RtpsRobot` in `strands_robots.mesh` wrap a ROS 2 graph as a mesh peer, so a rover on `/cmd_vel` appears in `peers` and answers `execute` and `stop` like any robot. Which to use: [ROS 2](../ros2.md).
