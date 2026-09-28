---
description: The operator dashboard - one process that joins the mesh, records datasets, trains and deploys policies, and puts a Strands Agent behind consent cards.
---

# Dashboard

One command, one tab, on the machine the robots are reachable from.

```bash
uv pip install 'strands-robots[dashboard,sim-mujoco]'
python -m strands_robots dashboard --open
```

It binds `127.0.0.1:8090` and opens the page; nothing reaches the network until
a passkey guards it.

## The first minute

1. **This machine, no passkey yet.** A browser on the same machine, at
   `http://127.0.0.1:8090` or `localhost`, is served; every other caller - a
   proxy or `ssh -L` forward, a foreign `Host` (DNS rebinding), another origin -
   is refused, because none of those is presence at the machine.
2. **Enrol the owner passkey.** The login screen asks for a bootstrap token: the
   `0600` file `enrol_token` beside `~/.strands_dashboard/auth.json`, or the
   `STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN` you set. Paste it, name the key, let the
   browser create the passkey. From then on every route but the login screen and
   `/api/health` answers `401` without a session.
3. **Bind a LAN address.** `--host 0.0.0.0` is refused until a passkey or a
   static `DASHBOARD_AUTH_TOKEN` exists, and says so.

```bash
python -m strands_robots dashboard --host 0.0.0.0 --port 8090
```

## What is on the page

The page is the operator SPA (React, built and committed under
`strands_robots/dashboard/static/`; no node at runtime). The process joins the
Zenoh mesh as a robot-less gateway, so one page drives hardware, simulators, or
a mix.

| Tab | What it shows | Where the rules live |
|---|---|---|
| Fleet | every live mesh peer with its joints, cameras, task and lockout state, plus every robot the registry knows; teleop pairing and a per-robot task form | `dashboard.routes_mesh`, `dashboard.mesh_bridge`, `dashboard.peer_tools` |
| Devices | the serial ports and cameras on this machine; spawn a robot process for a port (it joins the mesh as a managed child), assign cameras, read its log, despawn it | `dashboard.routes_devices`, `dashboard.device_manager`, `dashboard.bus_claim` |
| Record | a LeRobot dataset session: arms and cameras, start / stop / redo / discard episodes, thumbnails, labels, close and optionally upload | `dashboard.routes_record`, `dashboard.record_api` |
| Train | datasets and trainers, a graded job form, live loss, checkpoint search, validation against a robot, a deploy snippet | `dashboard.routes_train`, `dashboard.training`, `dashboard.checkpoints`, `dashboard.deploy` |
| Calibrate | the LeRobot calibration wizard for an arm, with the port owner and a confirm before the arm moves | `dashboard.calibration_run` |
| Sim | a MuJoCo robot stepping in this process - an MJPEG stream and the same model in your browser. Or a **mirror**: that twin posed from the real arm's servo bus, never written | `strands_robots.simulation`, `dashboard.mirror` |
| Agent | a Strands Agent over the fleet and the simulations; anything that moves a robot pauses on a consent card. The microphone opens voice | `dashboard.agent_console`, `dashboard.agent_hitl`, `dashboard.voice` |
| Settings | agent model and prompt, mesh endpoints, voice provider, editable `.env` keys, static token (shown only as set / unset) | `dashboard.settings`, `dashboard.config_api` |

Every path a client names must sit under its home (`HF_LEROBOT_HOME`,
`STRANDS_TRAIN_OUTPUT_DIR`, the Hub cache); anything else is refused without
saying whether the path exists.

## Two e-stops

`/api/safety/estop` stops the simulations this process runs;
`/api/mesh/safety/estop` is the signed fleet stop, whose answer says how many
peers replied and which did not (`responses_received`, `peers_not_stopped`), so
a stop that reached nobody never reads like one that reached everyone. The page
fires both when the mesh is online.

## The Agent tab

Type a sentence; the agent answers with tool calls you can read. Its tools are
the simulations - list, start, read joints, move, reset, stop, e-stop - through
the safety object the buttons use, so a latched e-stop refuses the agent as it
refuses a click, and stopping is never refused.

Moving a robot pauses first. `sim_set_joints` raises an interrupt before it runs,
the page shows what a yes would move (`2 - 1.000 rad`), and *Allow once*, *Allow
for this conversation* or *Refuse* resumes the same turn. A conversation-wide yes
lives in the socket and dies with it; every answer is audited. The model is the
one named by `STRANDS_MODEL_ID`, and the page shows which it is.

## The e-stop

The red button posts `/api/safety/estop`: every session stops stepping but keeps
rendering, so the robot stays on screen where it stopped, and the lockout latches
`locked`. While latched, any route that would move a sim answers `423` - a create
that overlapped the e-stop and a command already queued included; stopping never
does. `/api/safety/resume` lifts the lockout only to `unknown`: the next command a
session accepts is the proof. The same button reads RESUME while latched; its
label and its action are both read from the lockout the server last reported, so
a page opened under an e-stop engaged elsewhere shows RESUME. A refused request
is shown as a message, not painted as an e-stop.

## The twin follows the real arm

Pick a robot, change **simulate** to the serial port the arm is on (`GET
/api/sim/ports`, servo buses first) and press **Start**. The session is marked
**mirror · read-only**: a thread reads `Present_Position` from every motor at
~20 Hz and poses the model - no physics, no `Reset`, and `joints` answers `400`,
because the arm decides.

It never writes: the bus is closed with `disable_torque=False`, so torque stays
as you left it, and the footer says so. Angles are `(ticks - 2048) · 2π / 4096`
with no calibration applied, labelled `estimate` in the snapshot's `bus` field
with the raw ticks, read rate and age. A bus that stops answering shows
**stale**, then **error** with the reason; a pose the model refuses shows
**refused** with the joint named and clears when the arm comes back in range. A
port that will not open is a `502` naming it, and nothing is left holding it.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `STRANDS_DASH_AUTH_STORE` | `~/.strands_dashboard/auth.json` | the passkey store; auth is on once it holds a credential |
| `STRANDS_DASH_AUTH_ENABLED` | read from the store | force auth on (`1`) or off (`0`) |
| `STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN` | minted into `enrol_token` | the proof the first enrolment needs |
| `DASHBOARD_AUTH_TOKEN` | unset | a static bearer for scripts; removing a passkey still needs a passkey session |
| `DASHBOARD_SETTINGS_FILE` | `~/.strands_robots/dashboard/settings.json` | where Settings are written |
| `DASHBOARD_ENV_FILE` | `.env` | the file Settings writes env keys to |
| `DASHBOARD_JOBS_FILE` | `$TMPDIR/strands_dashboard/train_jobs.json` | the training job ledger |
| `STRANDS_MODEL_ID` | the SDK default | which Bedrock model the Agent tab talks to |
| `STRANDS_MESH` | on | `false` keeps this process off the mesh; Fleet shows the registry only |
| `HF_LEROBOT_HOME` | `~/.cache/huggingface/lerobot` | the dataset home every record / replay / label path sits under |
| `HF_HUB_CACHE` | the Hub default | where checkpoint search finds cached Hub snapshots |
| `STRANDS_ROBOTS_DATA_DIRS` | unset | extra dataset roots, colon separated, admitted next to the home |
| `STRANDS_TRAIN_OUTPUT_DIR` | `~/.strands_robots/training` | where training jobs may write |
| `STRANDS_DASH_AGENT_PHYSICAL_MOTION` | unset | the standing grant for real-hardware motion without a confirm; leave unset unless the operator is watching |
| `STRANDS_DASHBOARD_PROFILES` | `~/.strands_dashboard/profiles.json` | remembered USB device profiles (port, cameras, robot id) |
| `STRANDS_DASHBOARD_AUTOSPAWN` | on, off under pytest | `0` never respawns a remembered robot; `1` does so even under pytest |
| `STRANDS_DASHBOARD_SPAWN_SETTLE_S` | `5` | how long a spawn watches the new child before answering |
| `STRANDS_DASH_RECORD_CRUMB` | `~/.strands_dashboard/record_session.json` | the crumb a recording session leaves so a crash is reported next start |
| `VOICE_PROVIDER` / `VOICE_NAME` | `openai` | the `/ws/voice` provider (`openai`, `nova_sonic`) and its voice |
| `VOICE_MODEL` | the provider default | the provider model id for voice |
| `OPENAI_API_KEY` | unset | the `openai` voice key; Nova Sonic uses AWS credentials |
| `DASHBOARD_VOICE_PROMPT` | built in | replaces the voice system prompt |

The auth duration knobs (`STRANDS_DASH_AUTH_TOKEN_TTL`, `SESSION_MAX_AGE`,
`HANDOFF_TTL`) are in the [configuration reference](configuration.md).

## See also

- [Security](security.md) - the threat model.
- [Mesh](mesh.md) - how a fleet e-stop reaches every peer.
