---
description: An operator's command reaches one robot as an AWS IoT Core direct message; offline is one round trip.
---

# Direct messaging

By the end of this page an operator's command reaches one robot as an [AWS IoT Core direct message](https://docs.aws.amazon.com/iot/latest/developerguide/direct-messaging.html), the reply comes back the same way, and an offline robot is reported in one round trip.

## What changes

On the `iot` and `bridge` backends, `Mesh.send` makes one HTTPS call that delivers the command to the client it names, with confirmation (QoS 1 and the robot's PUBACK). The robot replies with a direct message on the `responseTopic` it received. No subscription is needed; a robot that is not connected answers `peer offline (iot 404)` at once, not after the caller's timeout. Measured 2026-09-29 in us-west-2: round trip p50 220 ms, offline verdict under 300 ms. `broadcast`, presence, state and safety stay publish/subscribe, and the `cmd` and `response/**` subscriptions stay, so an older peer is still heard.

## Grants

| statement | policy | topic condition |
|---|---|---|
| `AllowDirectCommandToAnyRobot` | operator | `strands/*/cmd` |
| `AllowDirectResponseToAnyOperator` | robot | `strands/*/response/${iot:Certificate.Subject.CommonName}/*` |

The robot grant reads the certificate subject: an HTTPS call has no MQTT connection for `${iot:Connection.Thing.ThingName}` to resolve from, so `provision_robot` issues certificates from a local CSR with `CN=<thing>`. A robot provisioned earlier (CN `AWS IoT Certificate`) gets 403 on its first direct reply, logged once, and answers over publish until it is re-provisioned. `bootstrap_account` adds the IAM policy `strands-operator-direct` for agents that command with credentials instead of a cert (`STRANDS_IOT_DIRECT_AUTH=sigv4`).

## Switches

`STRANDS_MESH_IOT_DIRECT=0` turns direct messaging off. `strands-robots doctor` has an `IoT Direct` row that sends this peer a confirmed message to itself and names the certificate CN on a 403.
