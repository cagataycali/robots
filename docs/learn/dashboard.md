# Dashboard

At the end of this page `strands-robots dashboard` is serving on your machine, you know what it shows at this commit and what it does not yet, how it decides who may click, and what its e-stop button does.

```bash
pip install 'strands-robots[dashboard,sim-mujoco]'      # fastapi, uvicorn, webauthn, PyJWT + MuJoCo for the twin
strands-robots dashboard --open                          # http://127.0.0.1:8090
```

Flags: `--host` (default `127.0.0.1`), `--port` (default `8090`), `--open`, `--log-level`. Binding a non-loopback host is refused until something guards the API: an enrolled passkey or a static `security.auth_token` in settings.

## What it serves

| surface | path | does |
|---|---|---|
| health | `GET /api/health` | process up, version; public |
| fleet | `GET /api/fleet`, `GET /api/robots/{name}` | every registry robot with category, joints, sim asset present on disk (`model_local`, no download), hardware backend; plus the mesh peers this process has heard, or `"off"` |
| sim twin | `POST /api/sim {"robot": "so101"}`, `GET /api/sim/{id}`, `POST .../joints`, `POST .../reset`, `GET .../scene`, `GET .../stream.mjpg`, `DELETE /api/sim/{id}`, `WS /ws/telemetry/{id}` | a MuJoCo session per browser tab: joint sliders, an MJPEG camera stream, telemetry |
| mirror | `POST /api/sim {"robot": "so101", "mirror": {"port": "/dev/ttyACM0"}}`, `GET /api/sim/ports` | the twin follows a real serial bus read-only (`BusMirror`); `ports` lists candidates and opens nothing |
| safety | `GET /api/safety`, `POST /api/safety/estop`, `POST /api/safety/resume` | the lockout as this dashboard knows it |
| agent | `WS /ws/agent`, `GET /api/agent` | one operator conversation with a Strands agent whose tools are the sim sessions |
| settings | `GET`, `POST /api/settings` | persisted settings; `security.auth_token` is reported as set or unset, never returned |
| auth | `/api/auth/status`, `register/*`, `login/*`, `logout`, `renew`, `handoff`, `credentials` | passkeys |

The UI is plain static files served by the same process: no build step, works offline.

Not at this commit: device spawning for real hardware, recording, training, calibration wizards and the live fleet view over the mesh. The dashboard reports peers the process already knows and does not join the mesh by opening a page; joining is `STRANDS_MESH`'s decision.

## Who may click

One dependency, `access.require_session`, guards every route except the login screen and `/api/health`. Three ways in, first match wins:

1. A passkey session token, as `Authorization: Bearer` or the `strands_dash` cookie. Never a query string (it would land in access logs).
2. The static `security.auth_token` from settings, compared in constant time.
3. Nothing, but only while no passkey is enrolled and the request is this machine's browser at this machine: loopback socket, no proxy header, a loopback `Host`, and an `Origin` naming the same host. A DNS-rebound page or a cross-site fetch arrives on loopback with the wrong `Host` or `Origin` and is refused.

Enrolling the first passkey closes the third door. That first enrollment must present `STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN`, or the token the process minted into a `0600` file beside the credential store; a loopback connection is not proof of presence at the machine. `STRANDS_DASH_AUTH_ORIGIN` and `STRANDS_DASH_AUTH_RP_ID` pin the WebAuthn origin and relying party behind a proxy; `STRANDS_DASH_AUTH_TOKEN_TTL` (86400 s), `..._SESSION_MAX_AGE` (2592000 s) and `..._HANDOFF_TTL` (300 s) bound a session, and a value that is not a whole number of seconds is refused rather than defaulted, because every substituted default is the wider one.

## The e-stop button

`POST /api/safety/estop` stops every sim session in this process and sets the lockout to `locked`. Every route that would move a sim then answers `423` until the lockout clears. `POST /api/safety/resume` sets the state to `unknown` on purpose: a resume is a request, not proof. The first command a session accepts afterwards is the proof, and only then does `GET /api/safety` say `clear`.

The mesh-wide signed e-stop ([safety and e-stop](mesh/safety-and-estop.md)) is a separate rail. This button stops what this dashboard runs; a mesh e-stop that reaches this process engages the same lockout.

## The agent in the browser

`/ws/agent` takes `{"type": "say", "text": ...}` and streams the console's events back (text, tool_use, tool_result, interrupt, done, error). Its tools go through the same `Safety` object the HTTP routes use, so the e-stop refuses the agent exactly as it refuses a button. `sim_set_joints` raises the same interrupt the real-hardware hook uses (`MotionInterruptHook`, see [agents](agents.md)); the browser shows a consent card and `{"type": "resume", "id": ..., "approve": true, "always": false}` resumes the same turn. `always` grants the rest of the conversation and dies with the socket. One turn at a time per socket; a second `say` is refused, not queued.

Two switches are off by default and matter only once a physical peer is reachable: `STRANDS_DASH_AGENT_PHYSICAL_MOTION=1` lets the agent's tools move a peer that is metal at all, and `STRANDS_DASH_TASK_REQUIRES_CONFIRM=1` makes a real-motion task POST carry an explicit boolean confirmation (a string is refused, not honoured). A simulated peer is never in their way.

## Logs

Everything the dashboard logs passes `log_redaction`: tokens, cookies and credential ids are masked before they reach a handler.
