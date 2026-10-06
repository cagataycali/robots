---
description: What emergency_stop() does to every robot it reaches, why the fleet stays stopped, and what a resume must prove.
---

# Safety and e-stop

At the end of this page you know what one `emergency_stop()` does to every robot it can reach, why a stopped fleet stays stopped until the operator's key says otherwise, what a resume must prove, and where each event is written down.

```python title="sketch"
# Every peer holds STRANDS_MESH_RESUME_PUBLIC_KEY; only the operator holds the signing key.
responses = a.mesh.emergency_stop()                 # stops the local robot, then broadcasts {"action": "stop"}
print([r["responder_id"] for r in responses])
print(a.mesh.send("arm-b", {"action": "execute", "instruction": "wave", "policy_provider": "mock"}))
# {'type': 'error', 'error': 'command rejected', ...}  deliberately generic
key = load_signing_key("resume_key.pem", passphrase)  # strands_robots.mesh.resume_authority
print(a.mesh.resume(key))                           # signs for this lockout, names a and its peers
# {'status': 'ok'}  or  {'status': 'error', 'error': 'resume rejected'}
```

No runnable fence on this page, on purpose: an e-stop reaches every peer the session can see, including a dashboard or a robot someone else left running. Run it when you mean it.

## What an e-stop does

{{drawing:d07_estop}}

1. Engages the local lockout and records the time.
2. Stops the robot in this process through the same `_dispatch({"action": "stop"})` path a remote peer would run. `broadcast` never reaches its sender, so without this step the fan-out misses the robot the operator stands next to.
3. Broadcasts `{"action": "stop"}` and collects replies for 3 s.
4. Publishes the event on `strands/safety/estop` (fleet-wide lockout) and `strands/<peer>/safety/event`.
5. Writes `emergency_stop` to the audit log.

Three stops, two resumes:

```mermaid
sequenceDiagram
    participant O as operator
    participant A as arm-a
    participant B as arm-b
    O->>A: emergency_stop()
    Note over A: lockout, stop
    A->>B: broadcast stop
    B-->>A: stopped
    A->>B: strands/safety/estop
    Note over B: lockout
    O->>B: resume, signed assertion
    B-->>O: ok
    B->>A: strands/safety/resume, same assertion
    Note over A: lockout cleared
```

The return value is every reply, the local one first. A reply counts as "stopped" only if it says so: a peer whose robot exposes no `stop_task` answers `{"ok": False}`, is logged at CRITICAL and listed in `peers_not_stopped`, never counted as halted. A mesh that is not running raises `RuntimeError` rather than returning `[]`: "asked nobody" cannot look like "asked, nobody answered".

## The lockout

While engaged, a peer answers only `status`, `ping`, `resume` and `stop`. `stop` stays admitted because it only de-energises: a second e-stop reaching a locked peer must still halt a rollout the first missed. Every other action is refused, and audited.

A peer receiving `strands/safety/estop` from another peer engages its own lockout; the log line reads `lockout engaged via remote estop from <issuer>`.

Replay defence there: the envelope's `t` must be fresh (`STRANDS_MESH_RESUME_FRESHNESS_S`, default 60 s), a per-receiver cache refuses a repeated `t`, issuers are capped per window. Refusing a cache slot never refuses the stop, nor does a receiver clock behind the operator: an estop up to a window early latches and audits `estop_clock_skew`; only `resume` keeps the forward-skew rule (5 s).

## Resume

The operator signs a resume. Create the key once with `python -m strands_robots.mesh.resume_authority keygen <key-file>` (passphrase: 16 characters, three classes) and set the printed `STRANDS_MESH_RESUME_PUBLIC_KEY` on every peer. Peers hold only that public half, so no peer can mint a resume. Without it a peer refuses every resume and says so at start:

```text
[safety:arm-a] No resume verification key set. If any peer broadcasts an e-stop, this robot stays locked until you physically restart it (one message can freeze the whole fleet).
```

The assertion names the lockout epoch (`Mesh.lockout_epoch`, the `estop_id` every peer one e-stop reached shares) and the target peers, so it clears no other robot and no later lockout. `Mesh.resume(key)` signs for its own lockout; `{"action": "resume", "assertion": ...}` carries one to a peer; a body carrying `override_code` is refused and audited. Refusals throttle after `STRANDS_MESH_RESUME_MAX_FAILS` (default 5) for `STRANDS_MESH_RESUME_BACKOFF_S` (default 30 s). The wire answer is `{"status": "ok"}` or `{"status": "error", "error": "resume rejected"}`; the reason stays in the local audit log.

A receiver refuses an assertion or envelope older than `STRANDS_MESH_RESUME_FRESHNESS_S` (default 60 s; a receiver whose clock is ahead of the operator trips it) and one past `STRANDS_MESH_RESUME_FORWARD_SKEW_S` (default 5 s, the tight one) in its future, tripped by a receiver behind the operator.

On success the peer relays the assertion on `strands/safety/resume`; each named peer checks signature, epoch, freshness and an unused nonce, then clears. A refusal leaves the lockout engaged.

## Stopping is never gated

On every surface, stop always runs: the robot tool's `stop`, `robot_mesh(action="stop")`, the mesh `stop` command under lockout, the dashboard's e-stop button. `emergency_stop` from the `robot_mesh` tool asks the operator first because it is fleet-wide and prompt-injectable; the rate limit there is 3 per minute.

## Audit

Every event on this page is one JSONL row in `~/.strands_robots/mesh_audit.jsonl` (`STRANDS_MESH_AUDIT_DIR`): `emergency_stop`, `remote_estop_engaged`, `estop_replay_rejected`, `estop_corroborated` (two operators within 0.2 s), `resume_ok`, `resume_denied` with the structured reason, `remote_resume_applied`, `remote_resume_redundant`, `command_rejected_lockout` for a command that arrived under lockout, `command_refused` for one a handler answered with an error. Rows carry `ts`, `event`, `peer_id`, `payload`, a process-monotonic `seq` so a deleted row shows as a gap, and `sig` (HMAC-SHA256) when `STRANDS_MESH_AUDIT_PSK` is set. `verify_audit_integrity()` walks the file and reports broken signatures, sequence gaps and unsigned rows. Rotation: `STRANDS_MESH_AUDIT_MAX_BYTES`, `STRANDS_MESH_AUDIT_MAX_FILES`.

See [security](../security.md) for the whole posture on one page.
