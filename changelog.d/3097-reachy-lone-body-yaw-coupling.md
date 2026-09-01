### Fixed: a Reachy Mini body-only turn is held to the head-body coupling limit

`HEAD_BODY_YAW_DELTA_LIMIT_DEG` bounds `head_yaw - body_yaw`, and
`envelope_error` only reached it when one action carried both names. Every motion
verb that sends one member routed around it: `reachy_body_turn` sends `body_yaw`
alone, `reachy_look` omits `body_yaw` at its `None` default, and an action naming
a head axis other than the yaw (`{"head_pitch": 10, "body_yaw": 160}`) commands
the head yaw to zero without spelling it, so the gate did not see that pair
either. `reachy_body_turn(yaw=160)` against a head at 0 reported `success` for a
body that reaches 65 degrees and stops.

The limit is the daemon's own. Its default kinematics solves a head pose through
`inverse_kinematics_safe(pose, body_yaw, max_relative_yaw=65 deg,
max_body_yaw=160 deg)` - the two figures this envelope carries - and it never
refuses an over-twist: it keeps the twist inside the limit by moving the body,
holding the head pose as the primary task. So an out-of-limit request does not
fail, it succeeds with a body yaw the caller did not ask for, which is the silent
substitution the envelope exists to refuse.

That makes the two directions different events, and only one of them is a defect.
A lone `head_yaw` of 180 is honored - the body turns to 115 under it, nothing of
the caller's is substituted, and refusing it would refuse the head verb its own
range. A lone `body_yaw` of 160 against a head target of 0 is replaced by one 95
degrees short. `ReachyDriver.send_action` now checks the coupling on a body-only
turn as well as on a pair, against the head yaw it last commanded - which is
exactly what the daemon is still targeting, and known rather than estimated
because the head command is a whole pose every time. The bound follows that
target rather than being a fixed range, so a body turn to 160 with the head
already round at 100 is 60 degrees of twist and still legal.

Unknown is not guessed. Before any head pose is commanded, and after
`play_move`, `wake_up`, `goto_sleep` or `set_motors` - the daemon re-pins its own
head target to wherever the head physically is when torque returns - the target
is forgotten and the coupling is skipped, because a turn refused against a stale
target is a turn the robot could have made.

`reachy_look` and `reachy_body_turn` said otherwise in the descriptions a model
reads back out of its own schema: `reachy_look` promised a refusal "beyond 65
deg" that its default path could not produce and said `body_yaw=None` "leaves the
body alone" when a large head yaw turns the whole robot, and `reachy_body_turn`
offered the full +/-160 with no mention of what bounds it. Both now state the
coupling and how to ask for a bigger turn. The Device Connect driver's
`_reject_unusable` no longer hands the pairwise limit to "`send_action`, which
takes both": one member is enough where the counterpart is known, and that
surface keeps no such record, so its `body` RPC stays per-axis only - graded now
rather than asserted in prose.
