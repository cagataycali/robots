### Fixed: a one-shot HITL motion grant names the call the human approved

`MotionInterruptHook` deposits a grant when a human approves a physical motion,
and that grant is spendable by exactly one call - so its key has to name that
call. It was read from `action` / `target` / `instruction`, three fields that are
constant for the two tools this layer is the *only* human gate for: `pose_tool`
and `serial_tool` declare no `target` (their peer is the `port`, which is why
`_resolve_target` reads that field instead) and carry the motion in fields rather
than in an instruction string. Every call of one action therefore hashed to the
same `tool|action||`. Measured over every gated action in `MOTION_ACTIONS`, 18 of
20 calls sat on a shared key, and a yes for `motor_name=shoulder_pan
position=2048` on `/dev/ttyACM0` was spendable by `motor_name=elbow_flex
position=4095` on `/dev/ttyACM1` - a different joint, on a different arm, to a
different angle, with no human asked.

The gate had already resolved the port and shown the operator those very fields;
the key was the one place that dropped them. `_grant_key` now reads the same
facts `motion_intent` resolves, the same way it reads them: the tool, the action
as the gate matched it (stripped), the target `_resolve_target` resolved, the
instruction, and the call's own motion fields. It is the `repr` of a tuple rather
than a `"|"` join, because these values are model-authored and a `"|"` inside one
of them would otherwise shift a boundary and let two different calls agree. A
per-build binding is not consulted and does not need to be: a bound proxy tool
IS its peer, so the tool name already names the robot.

`_DETAIL_FIELDS` was incomplete, which is the same root cause - the roster is
what makes one call distinguishable, and `motor_id`, `velocity` and `hex_data`
were absent from it. Those are the whole payload of `serial_tool`'s
`feetech_velocity` and the second spelling of `send` / `send_read`, so three
gated actions rendered as an empty detail line: the operator was asked to approve
a raw bus write and shown nothing about what went on the wire. `duration` joins
for the same reason on the `fleet` surface, since a yes for a five-second task
was otherwise spendable by a ten-minute one. The roster now has a single reader,
`_motion_fields`, so the line the operator reads and the identity their yes is
recorded against cannot come to describe a call differently.
