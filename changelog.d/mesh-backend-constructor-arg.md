# Robot(mesh_backend=...) - transport backend as constructor argument

`Robot("so100", mesh_backend="iot")` now swaps the mesh transport at the call
site, in addition to the existing `STRANDS_MESH_BACKEND` environment variable.
`init_mesh(robot, ..., mesh_backend="bridge")` accepts the same keyword when
constructing a mesh directly.

The constructor argument is a **call-site override**, not a replacement of the
env-var contract:

* When set (e.g. `mesh_backend="iot"`), it wins over any `STRANDS_MESH_BACKEND`
  value in the process environment for the duration of that construction.
* When left at its `None` default, the env var resolution is unchanged, so
  existing deployments that set `STRANDS_MESH_BACKEND` see today's behaviour
  byte-for-byte.
* An unknown value raises `ValueError` immediately at push time. Env-var typos
  still fall back to `zenoh` with a report (the historical policy that keeps
  the mesh running rather than crash the host); a caller-side typo we can
  name at the call site is a caller mistake we refuse instead of silently
  degrading.

The override is a `ContextVar`, so concurrent `Robot(mesh_backend=...)` calls
on different threads or asyncio tasks pick independent backends without
stamping over each other's choice.

Motivation: consistency with the rest of the `Robot()` API. Every other knob
(`backend=`, `mode=`, `driver=`, `mesh=`, `peer_id=`, `cameras=`) is a
constructor argument; the transport backend was the last piece that could only
be set via env var. It also unblocks per-call transport selection for fleets
that run one process talking to multiple backends (a bridge peer stitching
Zenoh LAN and AWS IoT, for example) without the env-var stomp that a single
process cannot express.
