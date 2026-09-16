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

The latch holds against a command that was already in flight. A route admits a
joints or reset request while the lockout is clear, the worker applies it a tick
later, and the red button can be pressed in between: folding that command in as
`note_command_accepted` reported the lockout `clear` while every session sat
frozen, and the next create was admitted. The check and the fold now happen
under the lock the e-stop latches under, so such a request answers 423 and the
latch stands. A queued command that would move the robot is refused when the
worker reaches it rather than applied, which is the promise `sim_session` states.

Ready means the engine built *and* rendered once, because creating the GL
context is the slow, machine-dependent half of starting a session - longer than
the model compile, and longer still under software GL. Paying it before the
create route answers means the first frame is already in the snapshot the caller
receives, and a machine with no renderer reports `error` there rather than as a
session that never streams.

That wait is bounded, and the bound is now answered rather than assumed. The
state published before the first frame is `running`, so a renderer that never
returns was handed back as a robot the operator could watch - 201, `running`,
nothing on the MJPEG stream, and one of the four session slots held for the life
of the process. The route reads what `wait_ready` reports: a session that misses
the budget is dropped and refused with 504, naming the robot and the budget, and
the slot it held is free again.

`/api/sim/{id}/joints` states its domain as `utils.finite_number_error`'s - the
one every surface that carries a signed physical quantity to a robot shares
instead of restating, the same ingress `3242-settings-non-finite-numeric-domain.md`
documents for the settings store. Restating it here admitted two values a
request body carries and `json.loads` builds without complaint: `true`, because
a `bool` is an `int` and `math.isfinite(True)` is True, so a checkbox posted as
a position was a 1 rad target the engine took; and an integer past the float64
range, which is finite and has no float form, so `math.isfinite` raised
`OverflowError` out of the guard written to answer rather than raise - a 500 for
the caller. Both answer 400 now, and the reason names the joint, or the list
index, that carried the value.

A joints request is a target, so the pose it writes survives the steps that
follow. The engine's `set_joint_positions` is a kinematic `qpos` write, and on a
robot held by position servos - `so101`, the one the page defaults to - the
servos are still commanded to their previous setpoint, so the worker's next
`step` pulls the pose back: a `0.5` rad target on joint `2` answered `200` and
read `0.03` rad half a second of sim time later. The session now writes with
`hold=True`, which moves the servo setpoints with the pose, and a real-engine
cell reads the joint back after that half second.
