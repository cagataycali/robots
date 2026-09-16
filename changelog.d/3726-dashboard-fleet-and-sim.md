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

Building an engine is not instant - a model compile plus a renderer - so a
session sits in `starting` for a moment, and an e-stop inside that window
reaches it too: `freeze_all` selects every session that can still step rather
than only the ones already running, and a create whose build overlapped the
e-stop is refused with 423 and dropped instead of being offered as the accepted
command that would report the lockout clear again.
