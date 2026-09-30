# Safety and e-stop

At the end of this page you know what one `emergency_stop()` does to every robot it can reach, why a stopped fleet stays stopped until an operator with the override code says otherwise, what a resume must prove, and where every one of those events is written down.

```python title="sketch"
# STRANDS_MESH_OVERRIDE_CODE must be the SAME value on every peer, set before Python starts.
responses = a.mesh.emergency_stop()                 # stops the local robot, then broadcasts {"action": "stop"}
print([r["responder_id"] for r in responses])
print(a.mesh.send("arm-b", {"action": "execute", "instruction": "wave", "policy_provider": "mock"}))
# refused: lockout engaged
print(a.mesh.send("arm-b", {"action": "resume", "override_code": "<the code>"}))
# {'status': 'ok'}  or  {'status': 'error', 'error': 'resume rejected'}
```

There is no runnable fence on this page on purpose. An e-stop reaches every peer the session can see, including a dashboard or a robot someone else left running on the same network. Run it when you mean it.

## What an e-stop does

1. Engages the local lockout and records the time.
2. Stops the robot in this process through the same `_dispatch({"action": "stop"})` path a remote peer would run. `broadcast` never comes back to its sender, so without this step the one robot the operator is standing next to would be the one the fan-out missed.
3. Broadcasts `{"action": "stop"}` and collects replies for 3 s.
4. Publishes the event on `strands/safety/estop` (fleet-wide lockout) and `strands/<peer>/safety/event`.
5. Writes `emergency_stop` to the audit log.

The return value is every reply, the local one first. A reply counts as "stopped" only if it says so: a peer whose robot exposes no `stop_task` answers `{"ok": False}`, is logged at CRITICAL and listed in `peers_not_stopped`. Counting it as an acknowledgement would tell an operator the fleet had halted while a robot was still moving. A mesh that is not running raises `RuntimeError` instead of returning `[]`, so "asked nobody" cannot look like "asked, nobody answered".

## The lockout

While engaged, a peer answers only `status`, `resume` and `stop`. `stop` stays admitted because it only de-energises: a second e-stop reaching a locked peer must still halt a rollout the first one missed. Every other action is refused and the refusal is audited.

A peer that receives `strands/safety/estop` from another peer engages its own lockout too. The log line reads `lockout engaged via remote estop from <issuer>`.

Replay defence on that topic: the envelope's `t` must be fresh (`STRANDS_MESH_RESUME_FRESHNESS_S`, default 60 s), a per-receiver cache refuses a repeated `t`, and issuers are capped per window. Refusing a cache slot never refuses the stop, nor does a receiver clock behind the operator: an estop up to a window early latches and audits `estop_clock_skew`; `resume` alone keeps the forward-skew rule.

## Resume

A resume is second-factor gated. `STRANDS_MESH_OVERRIDE_CODE` (at least 16 characters, say `secrets.token_urlsafe(32)`) must be configured on the peer that resumes and on every peer that honours it; a peer without one that long refuses every remote resume and says so at start:

```text
[safety:arm-a] No emergency-stop resume code set. If any peer broadcasts an e-stop, this robot stays locked until you physically restart it (one message can freeze the whole fleet).
```

`{"action": "resume", "override_code": ...}` on one peer compares the code in constant time, throttles after `STRANDS_MESH_RESUME_MAX_FAILS` (default 5) failures for `STRANDS_MESH_RESUME_BACKOFF_S` (default 30 s), and answers one of two shapes: `{"status": "ok"}` or `{"status": "error", "error": "resume rejected"}`. "Lockout not engaged", "code unconfigured" and "wrong code" all get the generic shape on the wire; the structured reason goes to the local audit log only, so a prober learns nothing about the fleet's state.

A receiver refuses a resume older than `STRANDS_MESH_RESUME_FRESHNESS_S` (default 60 s; a receiver whose clock is ahead of the operator trips it) and one past `STRANDS_MESH_RESUME_FORWARD_SKEW_S` (default 5 s, the tight one) in its future, tripped by a receiver behind the operator.

On success the peer publishes `strands/safety/resume` carrying an HMAC-SHA256 `override_proof` keyed with an scrypt-derived key over `peer_id`, `t`, `lockout_elapsed_s`, `proof_nonce` and the TLS session id, never the code itself. Receivers verify the proof under the same throttle, refuse a repeated `(issuer, proof_nonce)`, and only then clear their lockout. Every refusal leaves the lockout engaged.

## Stopping is never gated

On every surface, stop is the verb that always runs: the robot tool's `stop`, `robot_mesh(action="stop")`, the mesh `stop` command under lockout, the dashboard's e-stop button. `emergency_stop` from the `robot_mesh` tool asks the operator first because it is fleet-wide and prompt-injectable; the rate limit there is 3 per minute.

## Audit

Every event on this page is one JSONL row in `~/.strands_robots/mesh_audit.jsonl` (`STRANDS_MESH_AUDIT_DIR`): `emergency_stop`, `remote_estop_engaged`, `estop_replay_rejected`, `estop_corroborated` (two operators within 0.2 s), `resume_ok`, `resume_denied` with the structured reason, `remote_resume_applied`, `resume_replay_rejected`, `command_refused` under lockout. Rows carry `ts`, `event`, `peer_id`, `payload`, a process-monotonic `seq` so a deleted row shows as a gap, and `sig` (HMAC-SHA256) when `STRANDS_MESH_AUDIT_PSK` is set. `verify_audit_integrity()` walks the file and reports broken signatures, sequence gaps and unsigned rows. Rotation: `STRANDS_MESH_AUDIT_MAX_BYTES`, `STRANDS_MESH_AUDIT_MAX_FILES`.

See [security](../security.md) for the whole posture on one page.
