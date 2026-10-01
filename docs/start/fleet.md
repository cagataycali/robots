---
description: "Stage 5, a week: several robots on one mesh, a dashboard that sees them all, one e-stop that stops them all, and the postures that keep a stranger off the wire."
---

# Fleet

At the end of this rung two or more robots on two machines see each other on the mesh, an agent drives any of them through one tool with approvals in place, a dashboard shows the fleet live, and one e-stop stops every robot and keeps it stopped until an operator with the override code says otherwise.

{{drawing:d06_mesh_topology}}

## 1. Two peers on one machine

[Mesh](../learn/mesh/index.md). `STRANDS_MESH_LOCAL_DEV=true` and two `Robot(..., mesh=True)` in two processes: the first listens on `tcp/127.0.0.1:7447`, the second connects, each sees the other in `mesh.peers`. The mesh is enrichment; a session that fails to open leaves the robot working without it.

## 2. Command the fleet

[Fleet](../learn/mesh/fleet.md). `send` one peer an action, `tell` it a policy task, `broadcast` to all, `subscribe` to a peer's state topic. From an agent, the `robot_mesh` tool carries the same verbs with the gate in front of every one that moves a real robot. The wire vocabulary is a closed list; an action off it is refused before it is routed.

## 3. Second machine, real posture

[Mesh](../learn/mesh/index.md) again, postures. `ZENOH_CONNECT=tcp/<host>:7447` joins another host. Turn `STRANDS_MESH_LOCAL_DEV` off and the default posture is mTLS: the CA, certificate and key trio, refused when any is missing, and an ACL file with `default_permission: "deny"` for production. `strands-robots doctor` says which posture you are in and what it refuses.

## 4. Stop everything

[Safety and e-stop](../learn/mesh/safety-and-estop.md). `emergency_stop()` on any peer, or the dashboard's button, locks the local robot, stops it, broadcasts stop, publishes `strands/safety/estop` so every peer locks itself, and audits. Under lockout a peer answers only `status`, `resume` and `stop`. Set `STRANDS_MESH_OVERRIDE_CODE` (sixteen characters or more) before you need it: a peer without one stays locked until someone restarts it.

{{drawing:d07_estop}}

## 5. See it

[Dashboard](../learn/dashboard.md). Every peer as a card with its state, cameras and freshness, the consent card where the operator answers the gate, the e-stop that reaches the whole fleet. Behind a bridge, [bridges](../learn/mesh/bridges.md) relays the same topics to AWS IoT Core for a second site.

You now have a fleet: several robots on one wire, one tool that drives any of them with a person in the loop, and one stop that holds. Everything past this point is the reference: [tools](../reference/tools.md), [configuration](../reference/configuration.md), [refusal codes](../reference/refusal-codes.md).
