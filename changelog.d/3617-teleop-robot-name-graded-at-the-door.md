### Fixed: a teleop `robot_name` the host cannot route to is refused at the door

`teleoperate` and `start_teleop_receive` read `robot_name` only from *inside*
the loop they start - `send_action(merged, robot_name=...)` on every tick - which
is the category `teleoperate` already grades up front for `hz` and `duration`,
because "an unusable value used to be reported as a started session and only
misbehave on the background thread". `robot_name` was the member of that category
still ungraded, so a name that is not a robot in the world started a session
whose every frame the follower refused: in the default background mode the call
answered `success`, and nothing said otherwise until `stop_teleoperate` derived
`error` from the counters - by which time the leader had been connected and
polled for the whole session and the follower was never commanded.

`start_teleop_receive` is the sharper half. Its two mesh identifiers are
validated *ahead of* the teardown of any stream already registered under that
key, precisely so a refused call cannot stop a live one; `robot_name` skipped
that guarantee, so a typo stopped the receiver that was following this leader,
installed one the world refused on every frame, and reported success.

Both doors now ask the host once, through a new `TeleopMixin`
`_teleop_target_error` hook. A host that wraps exactly one device keeps the
documented accept-and-ignore behaviour (there is nothing to resolve); the MuJoCo
engine overrides it, reads the world the way its `send_action` does, and answers
in the same words - "Robot 'arm-2' not found. Did you mean: arm2? Available
robots: [...]" - so the door and the loop speak with one voice. `robot_name=None`
in a multi-robot world is still resolved by the loop rather than refused, and the
refusal lands before any device is connected, so it needs no rollback.
