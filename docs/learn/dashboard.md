---
description: What strands-robots dashboard serves, who may click, the two e-stops, and the agent console behind the consent card.
---

# Dashboard

At the end of this page `strands-robots dashboard` is serving, and you know what each tab does, who may click, and what the e-stops do.

```bash
pip install 'strands-robots[dashboard,sim-mujoco]'      # fastapi, uvicorn, webauthn, PyJWT + MuJoCo for the twin
strands-robots dashboard --open                          # http://127.0.0.1:8090
```

Flags: `--host` (default `127.0.0.1`), `--port` (default `8090`), `--open`, `--log-level`. A non-loopback host is refused until a passkey or static `security.auth_token` guards the API.

## What it serves

The process joins the Zenoh mesh as a robot-less gateway, under the same posture as your peers (`STRANDS_MESH_LOCAL_DEV=true` on one machine): one page drives hardware, simulators, or both, and every robot it shows or commands is a mesh peer. The UI is a built React SPA under `strands_robots/dashboard/static/`; no node at runtime. Each screen's rules live in the `strands_robots.dashboard` module named for it.

| tab | shows |
|---|---|
| Fleet | every live mesh robot (joints, cameras, task, lockout; a host process folds into its `__` robots) and every registry robot; teleop pairing, a task form |
| Devices | this machine's serial ports and cameras; spawn a robot process per port (a managed mesh child), assign cameras, read its log, despawn |
| Calibrate | the LeRobot calibration wizard, with a confirm before the arm moves |
| Agent | a Strands Agent whose tools are the fleet's peers; anything that moves a real robot pauses on a consent card; a microphone opens voice |
| Settings | agent model and prompt, mesh endpoints, voice provider, editable `.env` keys (a closed set; gates read-only), static token shown only as set / unset |

Every path a client names is resolved and must sit under its home (`HF_LEROBOT_HOME`, `STRANDS_TRAIN_OUTPUT_DIR`, the Hub cache); anything else gets one refusal that never reveals whether the path exists. A port must be a `/dev/...` path, a robot id one segment, a camera name `[A-Za-z0-9._-]`: each becomes a file name or argv.

Two e-stops: `POST /api/safety/estop` stops this process's simulations; `POST /api/mesh/safety/estop` is the signed fleet stop, whose answer names the peers that did not reply. The page fires both.

## Who may click

One dependency, `access.require_session`, guards every route but the login screen and `/api/health`. Three ways in, first match wins:

1. A passkey session token, as `Authorization: Bearer` or the `strands_dash` cookie; never a query string, which would land in access logs.
2. The static `security.auth_token`, compared in constant time.
3. The bootstrap token as bearer, only while no passkey is enrolled, from this machine's own browser: loopback socket, no proxy header, a loopback `Host`, an `Origin` naming the same host.

The first passkey closes the third door. Both need `STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN`, or the token the process minted into a `0600` file beside the credential store. `STRANDS_DASH_AUTH_ORIGIN` and `STRANDS_DASH_AUTH_RP_ID` pin the WebAuthn origin and relying party behind a proxy; `STRANDS_DASH_AUTH_TOKEN_TTL` (86400 s), `..._SESSION_MAX_AGE` (2592000 s) and `..._HANDOFF_TTL` (300 s) bound a session; a value that is not a whole number of seconds is refused, not defaulted.

Signing out or removing a passkey ends its sessions, open sockets included (re-checked every 2 s). Adding a passkey or removing the last needs the bootstrap proof.

## The e-stop button

`POST /api/safety/estop` stops every sim session in this process and locks; every route that would move a sim answers `423` until it clears. `POST /api/safety/resume` sets the state to `unknown` on purpose: a resume is a request, not proof; the first command a session accepts is, and only then does `GET /api/safety` say `clear`. The signed fleet stop ([safety and e-stop](mesh/safety-and-estop.md)) is a separate rail; its verdict is `fleet` in that answer.

## The agent in the browser

`/ws/agent` takes `{"type": "say", "text": ...}` and streams the console's events back (text, tool_use, tool_result, interrupt, done, error). The agent holds no robot of its own: `fleet` lists the peers, `spawn_robot` starts a registry robot in simulation as a new peer, and each peer is a tool named after it. Its `emergency_stop` shares the HTTP routes' `Safety` object, so the e-stop refuses the agent like a button. A motion verb on a real arm raises the real-hardware hook's interrupt (`MotionInterruptHook`, see [agents](agents.md)); the browser shows a consent card and `{"type": "resume", "id": ..., "approve": true, "always": false}` resumes the turn; `always` lasts the conversation and dies with the socket. One turn per socket; a second `say` is refused, not queued.

Two switches, off by default, matter once a physical peer is reachable: `STRANDS_DASH_AGENT_PHYSICAL_MOTION=1` lets the agent's tools move metal; `STRANDS_DASH_TASK_REQUIRES_CONFIRM=1` makes a real-motion task or teleop POST carry a boolean confirmation (strings are refused). Neither touches a simulated peer; both are granted and revoked from a consent card, never from Settings.

## Logs

Every log line passes `log_redaction`: tokens, cookies and credential ids are masked first.
