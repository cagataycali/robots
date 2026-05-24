# `target/` — the victim robot

A real `strands_robots.mesh.Mesh` running on a stub robot. The orchestrator
(`run_fleet.py`) provisions a fresh ephemeral CA + leaf cert per scenario,
builds the env, and forks this script. The rogue fires its attack against
the live `tls/127.0.0.1:<port>` endpoint.

## Why a real Mesh, not a stub

Every defence we want to verify lives behind real code paths:

| Defence                              | Lives in                                          |
| ------------------------------------ | ------------------------------------------------- |
| mTLS / tls-only link                 | `_zenoh_config.tls_block` + `link_protocols_block` |
| ACL shape validator (F2)             | `_acl_config._validate_acl_shape`                 |
| Wildcard `low_pass_filter` (F1)      | `_zenoh_config.low_pass_filter_block`             |
| Per-issuer estop replay cache        | `core.Mesh._on_safety_estop`                      |
| Resume-token HMAC + freshness        | `core.Mesh._on_safety_resume`                     |
| Audit-log HMAC + per-peer monotonic  | `audit.log_safety_event` / `verify_audit_integrity` |
| Payload validate_command             | `security.validate_command`                       |

Mocking any of those would invalidate the test.

## Process protocol

* **Ready signal**: a single line on stdout — `{"event":"READY","peer_id":...,"listen":"127.0.0.1:<port>","namespace":"strands","audit_dir":"..."}`.
  The orchestrator blocks on this line before launching the rogue.
* **Shutdown**: `SIGTERM` → graceful `mesh.stop()` → `{"event":"GOODBYE"}` on stdout.

## Stub robot surface

```python
class StubRobot:
    def get_task_status(self) -> dict: ...
    def get_state(self) -> dict | None: ...
    def execute(self, instruction: str, **kw) -> dict: ...
    def start(self, **kw) -> dict: ...
    def stop(self) -> dict: ...
    def reset(self) -> dict: ...
```

Returns idempotent OK responses — we are *not* testing the policy or robot
layers; the rogues stay strictly above the wire and at/below the dispatch
table.
