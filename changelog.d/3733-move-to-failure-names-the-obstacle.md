### `move_to` names what stopped the servo

A `move_to` that IK could solve but the servo could not reach used to end in
"the pose fights joint limits/contacts". It now reads the engine at the final
tick and says which: `The servo was stopped: the robot is in contact:
'ground' <-> 'so100/Fixed_Jaw/geom_18' (d=-0.0002 m)` (active contacts
touching the robot's own bodies, nearest first, at most three, total
reported) and/or `commanded joint(s) at a limit: 'arm/shoulder' at its upper
limit (1.0004 vs 1.0000)`. When neither is true it says so and points at
`max_steps`/`tol`. The same facts travel as `json.obstruction` (`contacts`,
`contacts_total`, `joints_at_limit`), with each contact spelled the way
`get_contacts` spells one (`geom1`, `geom2`, `dist`) so the two reports read
as one. On Isaac the joint half is reported and `contacts_total` is `null`
(contacts are not read there). Found by the v0.5.2 devx replay: three
failures in a row whose real cause was only visible in a separate
`get_contacts` call.
