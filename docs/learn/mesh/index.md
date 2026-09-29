---
description: Two robots see each other on the mesh; three switches decide whether it is on, how it is secured, which wire.
---

# Mesh

At the end of this page two robots in one process see each other on the mesh, one asks the other for its status and hands it a task, and you know the three switches: mesh on or off, how it is secured, which wire it rides.

No hardware needed. `STRANDS_MESH_LOCAL_DEV=true` is the single-machine preset (no TLS, no ACL, loud warnings); set it before the first `Robot(mesh=True)`.

```python
import os, time
os.environ.setdefault("STRANDS_MESH_LOCAL_DEV", "true")
from strands_robots import Robot

a = Robot("so101", mesh=True, peer_id="arm-a")
b = Robot("so101", mesh=True, peer_id="arm-b", tool_name="so101_b")
time.sleep(1.5)                                          # two heartbeats

print(a.mesh.alive, sorted(p["peer_id"] for p in a.mesh.peers))
# True ['arm-a__so101', 'arm-b', 'arm-b__so101']
print(a.mesh.send("arm-b", {"action": "status"}, timeout=5.0)["result"])
# {'status': 'idle', 'robots_running': []}
print(a.mesh.tell("arm-b", "wave", policy_provider="mock", duration=1.0)["result"]["status"])
# success
a.mesh.stop(); b.mesh.stop()
```

A simulation appears twice: the session peer (`arm-b`) and one child peer per robot in its world (`arm-b__so101`). If another mesh process is running on this machine, its peers show up too.

## What it is

Every `Robot` and every simulation can own a `Mesh`: a peer that broadcasts presence, publishes state and sensors, answers commands and relays teleoperation frames. The wire is Zenoh on the LAN (`[mesh]` extra, `eclipse-zenoh`), optionally bridged to AWS IoT Core for the cloud ([bridges](bridges.md)). Every message is JSON on a key like `strands/<peer>/state` ([topics](topics.md)).

The mesh is enrichment. A Zenoh session that fails to open leaves the robot working without it; the mesh never crashes the host.

## Three switches

| variable | values | effect |
|---|---|---|
| `STRANDS_MESH` | unset (default), `true`, `false` | `Robot(mesh=None)` follows this; `false` is a hard kill switch even over `mesh=True` |
| `STRANDS_MESH_AUTH_MODE` | `mtls` (default), `none` | `none` also needs `STRANDS_MESH_I_KNOW_THIS_IS_INSECURE=1` or `STRANDS_MESH_LOCAL_DEV=true` |
| `STRANDS_MESH_BACKEND` | `zenoh` (default), `iot`, `bridge` | which transport carries the topics; a typo falls back to `zenoh` and is reported once |

`Robot(..., mesh=True)` forces it on for one robot, `mesh=False` off. `init_mesh(robot, peer_id=...)` attaches one to any object with `send_action` and `stop`.

## Postures

| posture | set | for |
|---|---|---|
| local dev | `STRANDS_MESH_LOCAL_DEV=true` | one machine; multicast stays off, every process on the host meets at `tcp/127.0.0.1:7447` |
| trusted lab | `STRANDS_MESH_AUTH_MODE=mtls`, `STRANDS_MESH_TLS_CA`, `_TLS_CERT`, `_TLS_KEY`, `STRANDS_MESH_ACCEPT_PERMISSIVE_ACL=1` | any CA-signed peer may publish anywhere; `Mesh.start` refuses without the acknowledgement |
| production | the mTLS trio plus `STRANDS_MESH_ACL_FILE` with `default_permission: "deny"` | role-separated operators and robots ([bridges](bridges.md) covers the ACL file) |

Under `mtls` with no ACL file and no acknowledgement, `Mesh.start` refuses and prints the four ways out. `strands-robots doctor` prints the same text for the same posture.

The mTLS trio is required together: with any of the three unset or pointing at a missing file or a symlink, session open refuses with the variable names; the loader never downgrades to plain TCP. The key file must be mode `0600` on POSIX, checked on the real file. On Windows the mode check is skipped and the loader logs one WARNING per key file, so restrict the key with an NTFS ACL instead.

Discovery: the first process on a host listens on `tcp/127.0.0.1:<STRANDS_MESH_PORT>` (default 7447) and later ones connect to it, so every mesh process on one machine sees every other, a forgotten dashboard included. Across hosts set `ZENOH_CONNECT=tcp/10.0.0.1:7447` (comma-separated) or `ZENOH_LISTEN`. `STRANDS_MESH_MULTICAST=true` opens UDP `224.0.0.224:7446` so any device on the LAN can find your fleet; it is off by default and logs a warning when on.

## Rates and caps

Presence at 2 Hz, state at 10 Hz, camera off until `STRANDS_MESH_CAMERA_HZ` is set. Commands are capped at 20 Hz and 16 KiB per message, safety topics at 2 Hz and 4 KiB, camera frames at 1 MiB, sessions at 256. Each cap has a `STRANDS_MESH_*` override listed in [reference/configuration](../../reference/configuration.md).

## Pages

- [fleet](fleet.md): join, discover, `tell`, `send`, `broadcast`, RPC, the `robot_mesh` tool.
- [safety and e-stop](safety-and-estop.md): `emergency_stop`, the lockout, the override code, the audit trail.
- [topics](topics.md): every key and its rate.
- [bridges](bridges.md): the IoT and bridge transports, the ACL file.
- [direct messaging](direct.md): point-to-point commands over AWS IoT Core.
