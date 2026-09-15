### Added: the dashboard shows the fleet and runs a simulated robot behind the e-stop

`/api/fleet` reads the registry (`list_robots`) and, when the mesh extra is
importable, the in-process peer table - it never joins the mesh as a side
effect of opening a page. `/api/sim` starts `Robot(name, mode="sim")` in one
worker thread per session (the renderer's GL context is thread-bound and
`MjData` is not shareable), capped at four, and serves the camera as MJPEG
through `rendering.video.mjpeg_frames` plus a `/ws/telemetry` snapshot stream.

The e-stop is `safety_state.Lockout`, folded exactly as that module's tests
describe: `/api/safety/estop` freezes every session and latches `locked`;
every route that would move a sim checks `proves_clear` and refuses with 423;
`/api/safety/resume` leaves the state `unknown`, because a resume is a request,
and the first command a session then accepts is `note_command_accepted`, the
proof. Stopping a session is never refused.

### Added: a 3D twin in the browser, drawn from the compiled model

`/api/sim/{id}/scene` describes the geoms, meshes and cameras of the session's
`MjModel`; `/api/sim/{id}/mesh/{i}` serves each compiled mesh as bytes read
straight out of `mesh_vert`/`mesh_face` - no file is opened, so there is no
path to contain. With `?poses=1` the telemetry socket follows every JSON
snapshot with one binary frame of `geom_xpos|geom_xmat` rows, and
`static/twin.js` (three.js r170, vendored under `static/vendor/` with its MIT
notice so the page works on a LAN with no internet) sets each object's matrix
from it. No physics runs in the browser and no MJCF is parsed there: the twin
shows exactly what MuJoCo computed, for every robot the engine can load.

### Added: the Agent tab - a Strands Agent whose hands are the simulations

`/ws/agent` is one operator conversation with a Strands `Agent`
(`dashboard/agent_console.py`, model from `STRANDS_MODEL_ID`). Its tools
(`robots`, `sim_sessions`, `sim_start`, `sim_state`, `sim_set_joints`,
`sim_reset`, `sim_stop`, `emergency_stop`) go through the same `Safety`
object as the buttons, so the e-stop refuses the agent the way it refuses a
click and stopping is never refused. `sim_set_joints` raises the SDK
interrupt (`sim_motion`) before it runs; the browser shows a consent card
with exactly what a yes moves (`2 → 1.000 rad`), and *Allow once*, *Allow
for this conversation* or *Refuse* resumes the same turn. Grants live in the
socket and die with it; every answer is written to the HITL audit log.
