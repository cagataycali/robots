---
description: The operator dashboard - one process on your machine that shows the fleet, runs a simulated robot you can watch, and puts a Strands Agent behind consent cards.
---

# Dashboard

One command, one browser tab, on the machine the robots are reachable from.

```bash
uv pip install 'strands-robots[dashboard,sim-mujoco]'
python -m strands_robots dashboard --open
```

It binds `127.0.0.1:8090` and opens the page. Nothing is exposed to the network
until a passkey guards it.

## The first minute

1. **This machine, no passkey yet.** The dashboard is usable from a browser on
   the same machine, at `http://127.0.0.1:8090` or `localhost`, and refuses
   every other caller - a same-host proxy or `ssh -L` forward, a request whose
   `Host` is any other name (DNS rebinding), and a page from any other origin
   (cross-site `fetch` or `WebSocket`). None of those is presence at the
   machine.
2. **Enrol the owner passkey.** The login screen asks for a bootstrap token. It
   is in the `0600` file `enrol_token` beside `~/.strands_dashboard/auth.json`
   (or the `STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN` you set). Paste it, name the key,
   let the browser create the passkey. From that moment the API is sealed: every
   route but the login screen and `/api/health` answers `401` without a session.
   The login screen's own route says whether setup is required and which proof
   it needs - never the enrolled passkeys, which only a session may list.
3. **Bind a LAN address if you want to.** `--host 0.0.0.0` is refused until a
   passkey or a static `DASHBOARD_AUTH_TOKEN` exists, and says so.

```bash
python -m strands_robots dashboard --host 0.0.0.0 --port 8090
```

## What is on the page

| Tab | What it shows | Where the rules live |
|---|---|---|
| Fleet | every robot the registry knows, sim and real, and the mesh peers when the `[mesh]` extra is installed | `strands_robots.registry` |
| Sim | up to four MuJoCo robots stepping in this process, one thread each - the rendered camera as MJPEG and a 3D twin drawn from the compiled model, joint sliders, reset, stop. Or a **mirror**: the same twin posed from the real arm's servo bus, read and never written | `dashboard.sim_session`, `dashboard.scene`, `dashboard.mirror` |
| Agent | a Strands Agent whose tools are the simulations on the Sim tab; a move it wants to make pauses on a consent card until you answer | `dashboard.agent_console` |
| Settings | the file `~/.strands_robots/dashboard/settings.json` - agent model, mesh endpoints, static token (shown only as set / unset) | `dashboard.settings` |

Every string a route serves is rendered as text, never as markup: a Fleet row can
carry a mesh peer's name, and script running in this page would be same-origin -
it carries the session cookie and names this origin as its own, so it is behind
every guard above by construction. `tests/test_dashboard_static_renders_data_as_text.py`
reads that rule off the files the wheel ships.

## The e-stop

The **E-STOP** button on the Sim tab is `POST /api/safety/estop`. It is never
refused, never needs anything but a session, and does three things at once:
every simulation freezes, the dashboard's **lockout** latches (`/api/safety`
answers `{"lockout": {"state": "locked", "reason": ..., "by": ...}}`), and from
then on every command that would move something - creating a session, `reset`,
`joints`, the agent's motion tools - answers `423` until someone presses
**Resume**. Stopping a session and reading state are exempt: you can always
stop, you can always look. The lockout lives in `dashboard.safety_state`, the
same module the mesh e-stop uses, so a fleet stop and a dashboard stop are one
state, not two.

## The twin follows the real arm

Pick a robot, change **simulate** to the serial port the arm is on (the list
is `GET /api/sim/ports`, servo buses first) and press **Start**. The session
that appears is marked **mirror · read-only**: a thread reads
`Present_Position` from every motor at ~20 Hz and the model is posed from the
readings - no physics steps, no `Reset`, and `joints` answers `400`, because
the arm decides. Move the arm by hand and the twin moves.

What it will not do is write. lerobot's bus is opened for the handshake (a
ping and a firmware read) and closed with `disable_torque=False`, since the
default close writes `Torque_Enable=0` to every motor. Torque stays exactly as
you left it and the footer says so. Angles are `(ticks - 2048) · 2π / 4096`
with no calibration applied - right up to the offset a calibration would
record, and labelled `estimate` in the snapshot's `bus` field along with the
raw ticks, the read rate and the age of the last reading. A bus that stops
answering shows **stale**, then **error** with the reason; a port that will
not open is a `502` naming it, and nothing is left holding the device.

## Talking to it from a script

Everything the page does is a JSON route under `/api`, behind the same session
(a passkey cookie, or `Authorization: Bearer $DASHBOARD_AUTH_TOKEN`).

| Route | What |
|---|---|
| `GET /api/fleet` · `GET /api/robots/{name}` | the registry, sim and real, plus mesh peers when `[mesh]` is installed |
| `GET /api/sim` · `POST /api/sim {"robot": "so101"}` · `DELETE /api/sim/{id}` | list, start (`201`, `429` past four), stop |
| `POST /api/sim {"robot": "so101", "mirror": {"port": "/dev/cu.usbmodem…"}}` · `GET /api/sim/ports` | start a mirror of the real arm on that port (`400` if it is not a device here, `502` with the reason if the bus will not open); the ports a mirror could read |
| `GET /api/sim/{id}` | the snapshot: `state`, `sim_time`, `steps`, `joint_names`, `qpos` (radians), `fps`, `cameras`, `source` (`sim` or `real:<port>`) and, for a mirror, `bus` (`hz`, `age_ms`, raw `ticks`, `error`) |
| `POST /api/sim/{id}/joints {"positions": {"2": 1.0}}` · `POST /api/sim/{id}/reset` | move (joint name or 1-based index as the key) or go home; `423` under lockout |
| `GET /api/sim/{id}/stream.mjpg[?frames=N]` | the rendered camera, 12 fps multipart MJPEG; `frames` bounds a probe |
| `GET /api/sim/{id}/scene` · `GET /api/sim/{id}/mesh/{index}` | the compiled model's geoms (`type`, `size`, `rgba`, `body`, `mesh`) plus cameras and lights, and each mesh's vertices and faces as binary, for the twin |
| `WS /ws/telemetry/{id}[?poses=1]` | the snapshot at 15 Hz; with `poses=1` each JSON frame is followed by one binary frame of `ngeom × 12` little-endian `float32` world poses (3 position + 9 rotation) |
| `GET /api/safety` · `POST /api/safety/estop` · `POST /api/safety/resume` | the lockout, and the two ways to change it |
| `GET /api/agent` · `WS /ws/agent` | the agent's model and which tools ask first; the conversation |

### The agent socket

`/ws/agent` is one conversation. You send `{"type": "say", "text": "..."}`;
the dashboard streams back flat events until `{"type": "done"}`:

```text
{"type": "text", "text": "I'll start the so101 first."}
{"type": "tool_use", "name": "sim_start", "input": {"robot": "so101"}}
{"type": "tool_result", "status": "success", "text": "{\"id\": \"655a3be6\", ...}"}
{"type": "tool_use", "name": "sim_set_joints", "input": {"session_id": "655a3be6", "positions": {"2": 1.0}}}
{"type": "interrupt", "id": "...", "name": "sim_motion", "reason": {"session_id": "655a3be6", "detail": "2 → 1.000 rad", ...}}
```

An `interrupt` is the agent paused mid-turn by a Strands SDK interrupt - the
tool has not run. Answer `{"type": "resume", "id": "...", "approve": true,
"always": false}` and the same turn continues; `approve: false` cancels that
one call and the agent is told the operator declined. `always: true` stops the
card from coming back for that session for as long as the socket lives.
Anything but an explicit `true` is a no, and every answer is written to the
HITL audit log. The agent's tools are the sim routes above under the same
lockout, so it cannot do anything the page cannot; stopping is never gated.
The model comes from `STRANDS_MODEL_ID` (Bedrock), the same variable the CLI
reads; with no usable model the socket answers one `error` frame and closes.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `STRANDS_DASH_AUTH_STORE` | `~/.strands_dashboard/auth.json` | the passkey store; auth is on the moment it holds a credential |
| `STRANDS_DASH_AUTH_ENABLED` | read from the store | force on (`1`) or off (`0`); anything else is ignored with a warning |
| `STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN` | minted into `enrol_token` | the proof the first enrolment needs |
| `DASHBOARD_AUTH_TOKEN` | unset | a static bearer for scripts; a passkey session is still needed to remove a passkey |
| `DASHBOARD_SETTINGS_FILE` | `~/.strands_robots/dashboard/settings.json` | where Settings are written |

Every auth duration knob (`STRANDS_DASH_AUTH_TOKEN_TTL`, `SESSION_MAX_AGE`,
`HANDOFF_TTL`) is documented in the [configuration reference](reference/configuration.md);
none can be widened past its cap.

## See also

- [Security](security.md) - the threat model the dashboard is built against.
- [Mesh](mesh.md) - how a fleet e-stop reaches every peer.
