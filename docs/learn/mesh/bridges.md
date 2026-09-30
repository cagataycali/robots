# Bridges

At the end of this page your fleet's presence, commands and safety events reach AWS IoT Core over MQTT5 with per-robot X.509 identities while joint state and camera frames stay on the LAN, and you know how to write the ACL that separates operators from robots on the Zenoh side.

```bash
pip install 'strands-robots[mesh-iot]'          # awsiotsdk, awscrt, boto3 on top of [mesh]
```

```python title="sketch"
from strands_robots.mesh.iot import bootstrap_account, provision_robot

bootstrap_account()                                   # once per account and region: Rules, Lambda, DynamoDB audit, Fleet Provisioning template
p = provision_robot("so101-arm-01")                   # per robot: cert + policy + Thing named after the peer
print(p.env_vars())                                   # STRANDS_IOT_THING_NAME, STRANDS_IOT_ENDPOINT, STRANDS_IOT_CERT_DIR, STRANDS_MESH_BACKEND=iot
```

Export those variables, set `STRANDS_MESH_BACKEND=bridge`, and `Robot("so101", mode="real", mesh=True, peer_id="so101-arm-01")` publishes on both wires.

## Three backends

| `STRANDS_MESH_BACKEND` | transport | for |
|---|---|---|
| `zenoh` (default) | `ZenohTransport`: one LAN session per process, mTLS and ACL from `STRANDS_MESH_*` | a room, a lab |
| `iot` | `IotMqttTransport`: MQTT5 over mTLS to AWS IoT Core, no Zenoh | a robot whose only peers are in the cloud |
| `bridge` | `BridgeTransport`: one of each; every `put` fans out, subscriptions fan in | production: LAN peers plus operator dashboards, audit and fleet ops in AWS |

The bridge degrades rather than fails: Zenoh down means pure IoT, IoT down means pure Zenoh, both down means `is_alive()` is false and every put is a no-op.

## What the bridge forwards

The MQTT side is filtered by topic suffix. Default to both wires: `presence`, `health`, `cmd`, `response`, `broadcast`, `safety/event`, `safety/estop`, `safety/resume`. LAN-only: `state`, `pose`, `imu`, `odom`, `camera`, `input`, `hand`, `stream`. `STRANDS_MESH_BRIDGE_TOPICS` is a comma-separated suffix list that replaces the default. Inbound duplicates (a presence that arrived on both wires) are dropped at the `Mesh` layer by `sender_id` and `turn_id`; `STRANDS_MESH_BRIDGE_DEDUP_STRICT` tightens that.

Keys are unchanged on MQTT (`strands/<peer>/cmd` is a valid MQTT topic); wildcards map `*` to `+` and `**` to `#`.

## The IoT trust model

Each robot is a Thing whose name equals its mesh `peer_id` and its cert CN; the MQTT `client_id` is set to it so `${iot:Connection.Thing.ThingName}` in the IoT policy scopes every robot to its own topics. `provision_robot(thing_name, region=, cert_dir=, attributes=, allow_estop_publish=False)` generates the key locally, has AWS sign a CSR with `CN=<thing>, O=strands-robots`, and writes `<thing>.cert.pem`, `<thing>.private.key` and `AmazonRootCA1.pem` under `STRANDS_IOT_CERT_DIR` (default `~/.strands_robots/iot`). Re-running publishes a changed policy document as the new default version.

| variable | meaning |
|---|---|
| `STRANDS_IOT_ENDPOINT` | the account's ATS endpoint |
| `STRANDS_IOT_THING_NAME` | the Thing, equal to the cert CN and the peer id |
| `STRANDS_IOT_CERT_DIR` | where the three PEM files live |
| `STRANDS_IOT_CA_FILE` | overrides the root CA path |
| `STRANDS_IOT_DIRECT_AUTH` | `x509` (default with a cert) or `sigv4` (IAM credentials) for direct messages |

Missing `awsiotsdk`, endpoint or cert files make `connect()` return `False` with an ERROR line; the mesh stays off.

Cloud mirrors: `shadow.enable_for_mesh(mesh)` keeps a Device Shadow (`presence`) per Thing; `camera_offload.enable_for_mesh(mesh)` puts camera frames in S3 (`STRANDS_MESH_CAMERA_S3_BUCKET`, `STRANDS_MESH_CAMERA_S3_PREFIX`) and publishes presigned URLs (`STRANDS_MESH_CAMERA_PRESIGN_TTL`).

## The Zenoh ACL

Under `STRANDS_MESH_AUTH_MODE=mtls` the built-in ACL lets any CA-signed peer publish and subscribe anywhere, and `Mesh.start` refuses that posture until you set `STRANDS_MESH_ACCEPT_PERMISSIVE_ACL=1` or point `STRANDS_MESH_ACL_FILE` at a JSON5 file. The template is `examples/mesh/mesh_acl_example.json5`: `default_permission: "deny"`, named `rules`, `subjects` keyed by literal `cert_common_names`, and `policies` binding the two. Its three roles:

| subject | may |
|---|---|
| `robot_peer` | publish telemetry; subscribe to presence, broadcast, safety and its own `cmd` and `response` |
| `operator_peer` | publish commands; subscribe to `**` |
| `dashboard_peer` | subscribe to observe; publish presence |

Facts about Zenoh 1.x the loader has verified live: `enabled: true` is required or the block is a no-op; `cert_common_names` match literally (no globs), so you enumerate every CN; omitting `interfaces` matches every link and `[]` is rejected; `key_exprs` see the key without the namespace, so `**/cmd` works and `strands/*/cmd` matches nothing; `declare_subscriber` rules live in the `egress` flow and `put` rules in `ingress`. There is no `${cn}` interpolation, so strict per-peer isolation on Zenoh means one rule per robot CN (`mesh_acl_strict_per_peer.json5`); the IoT transport gives that isolation by construction. A file is one of two shapes: `default_permission: "deny"` plus `allow` rules is a whitelist, where a gap denies; `default_permission: "allow"` plus rules is a blacklist, where a key expression nobody named is open on the wire. `_parse_acl_bytes` (`mesh/_acl_config.py`) refuses the blacklist shape with `PermissiveACLError` unless `STRANDS_MESH_ACCEPT_PERMISSIVE_ACL` is `1`, `true` or `yes`. The same token has two more readers: `Mesh._refuse_under_permissive_default_acl` (`mesh/core.py`), the start gate under the built-in permissive default, and `session._build_config` (`mesh/session.py`), whose per-session WARNING that no ACL file is set is suppressed by it; `check_mesh` (`doctor.py`) reports all of this. So a token set to load a blacklist in CI also waives the start gate: drop the file later and the fleet runs wire-open with no log signal. Do not set it on a production fleet.

`STRANDS_MESH_CA_PINS` pins the CA fingerprints; `STRANDS_MESH_DISABLE_CA_PIN` turns pinning off for a lab.

## ROS 2 as a peer

`RosBridgedRobot`, `RosbridgeRobot` and `RtpsRobot` in `strands_robots.mesh` wrap a ROS 2 graph as a mesh peer, so a rover on `/cmd_vel` appears in `peers` and answers `execute` and `stop` like any other robot. Which one to use is on [ROS 2](../ros2.md).
