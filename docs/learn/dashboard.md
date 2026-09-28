# Dashboard

At the end of this page `strands-robots dashboard` is serving on your machine, and you know what each tab does, who may click, and what the e-stop buttons do.

```bash
pip install 'strands-robots[dashboard,sim-mujoco]'      # fastapi, uvicorn, webauthn, PyJWT + MuJoCo for the twin
strands-robots dashboard --open                          # http://127.0.0.1:8090
```

Flags: `--host` (default `127.0.0.1`), `--port` (default `8090`), `--open`, `--log-level`. A non-loopback host is refused until an enrolled passkey or a static `security.auth_token` guards the API.

## What it serves

The process joins the Zenoh mesh as a robot-less gateway, so one page drives hardware, simulators, or a mix. The UI is a React SPA committed under `strands_robots/dashboard/static/`; no node at runtime.

| tab | shows | rules live in |
|---|---|---|
| Fleet | every live mesh peer with joints, cameras, task and lockout state, plus every registry robot; teleop pairing and a task form | `routes_mesh`, `mesh_bridge`, `peer_tools` |
| Devices | this machine's serial ports and cameras; spawn a robot process for a port (a managed mesh child), assign cameras, read its log, despawn it | `routes_devices`, `device_manager`, `bus_claim` |
| Record | a LeRobot dataset session: arms and cameras, start / stop / redo / discard episodes, thumbnails, labels, upload | `routes_record`, `record_api` |
| Train | datasets and trainers, a graded job form, live loss, checkpoints, validation against a robot, a deploy snippet | `routes_train`, `training`, `checkpoints`, `deploy` |
| Calibrate | the LeRobot calibration wizard for an arm, with a confirm before the arm moves | `calibration_run` |
| Sim | a MuJoCo robot stepping in this process, streamed and rendered in the browser; or a **mirror**, the twin posed from a real arm's servo bus, never written | `strands_robots.simulation`, `mirror` |
| Agent | a Strands Agent over the fleet and the simulations; anything that moves a robot pauses on a consent card; a microphone button opens voice | `agent_console`, `agent_hitl`, `voice` |
| Settings | agent model and prompt, mesh endpoints, voice provider, editable `.env` keys (a closed set; gates are read-only here), static token shown only as set / unset | `settings`, `config_api` |

Every path a client names is resolved and must sit under its home (`HF_LEROBOT_HOME`, `STRANDS_TRAIN_OUTPUT_DIR`, the Hub cache); anything else gets one refusal that does not say whether the path exists. A port must be a `/dev/...` path, a robot id one segment, a camera name `[A-Za-z0-9._-]`: each becomes a file name or a child's argv.

Two e-stops: `POST /api/safety/estop` stops the simulations this process runs; `POST /api/mesh/safety/estop` is the signed fleet stop, whose answer names the peers that did not reply, so a stop that reached nobody never reads like one that reached everyone. The page fires both.

## Who may click

One dependency, `access.require_session`, guards every route except the login screen and `/api/health`. Three ways in, first match wins:

1. A passkey session token, as `Authorization: Bearer` or the `strands_dash` cookie. Never a query string (it would land in access logs).
2. The static `security.auth_token` from settings, compared in constant time.
3. Nothing, but only while no passkey is enrolled and the request is this machine's browser at this machine: loopback socket, no proxy header, a loopback `Host`, an `Origin` naming the same host. A DNS-rebound page or a cross-site fetch fails the `Host` or `Origin` check and is refused.

Enrolling the first passkey closes the third door. That enrollment must present `STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN`, or the token the process minted into a `0600` file beside the credential store; loopback alone is not proof of presence. `STRANDS_DASH_AUTH_ORIGIN` and `STRANDS_DASH_AUTH_RP_ID` pin the WebAuthn origin and relying party behind a proxy; `STRANDS_DASH_AUTH_TOKEN_TTL` (86400 s), `..._SESSION_MAX_AGE` (2592000 s) and `..._HANDOFF_TTL` (300 s) bound a session; a value that is not a whole number of seconds is refused, not defaulted, because every substituted default is the wider one.

## The e-stop button

`POST /api/safety/estop` stops every sim session in this process and sets the lockout to `locked`; every route that would move a sim then answers `423` until it clears. `POST /api/safety/resume` sets the state to `unknown` on purpose: a resume is a request, not proof. The first command a session accepts is the proof, and only then does `GET /api/safety` say `clear`. The signed fleet stop ([safety and e-stop](mesh/safety-and-estop.md)) is a separate rail; one that reaches this process engages the same lockout.

## The agent in the browser

`/ws/agent` takes `{"type": "say", "text": ...}` and streams the console's events back (text, tool_use, tool_result, interrupt, done, error). Its tools go through the `Safety` object the HTTP routes use, so the e-stop refuses the agent as it refuses a button. `sim_set_joints` raises the interrupt the real-hardware hook uses (`MotionInterruptHook`, see [agents](agents.md)); the browser shows a consent card and `{"type": "resume", "id": ..., "approve": true, "always": false}` resumes the turn. `always` lasts the conversation and dies with the socket. One turn per socket; a second `say` is refused, not queued.

Two switches, off by default, matter once a physical peer is reachable: `STRANDS_DASH_AGENT_PHYSICAL_MOTION=1` lets the agent's tools move metal at all, and `STRANDS_DASH_TASK_REQUIRES_CONFIRM=1` makes a real-motion task or teleop POST carry an explicit boolean confirmation (a string is refused). A simulated peer is never in their way. Both are granted and revoked from a consent card, never from the Settings tab.

## Logs

Everything the dashboard logs passes `log_redaction`: tokens, cookies and credential ids are masked before any handler.
