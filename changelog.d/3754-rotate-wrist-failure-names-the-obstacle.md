### `rotate_wrist` names what stopped the servo

A `rotate_wrist` that stopped short reported only its residual, so the cause
needed a separate `get_contacts` call - measured on the bundled so100, the
wrist turned 0.003 of 2.500 rad while 15 contact pairs held the gripper inside
its own base and the refusal named none of them. It now reads the engine at
the final tick through the same reader `move_to` uses and names it: `The servo
was stopped: the robot is in contact: 'arm/Base/geom_3' <->
'arm/Fixed_Jaw/geom_19' (d=-0.0456 m) and 12 more`, or `commanded joint(s) at
a limit: 'arm/wrist_roll' at its upper limit (1.0004 vs 1.0000)`, with the
same facts as `json.obstruction`. The report is scoped to the wrist joint:
this primitive holds every other joint where it already was, so a bound one of
those is not offered as the cause. On Isaac the wrist bound is reported and
`contacts_total` is `null` (contacts are not read there).

Because the clause is now shared by two primitives, its remedy names the
commanded value rather than a target position - `rotate_wrist` commands an
angle, and "move the target away" was not an instruction its caller could
follow.
