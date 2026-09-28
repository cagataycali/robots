### Fixed: dashboard Fleet shows a spawned sim robot once, and a confirmed resume clears its e-stop badge

A sim robot spawned from the Devices drawer is two mesh peers: the host process
`<host>` and the robot it holds, `<host>__<robot>`. The Fleet drew a card for
each, so the host appeared as a second robot with no joints and its own Run
button; it is now folded into its robot's card.

The robot's `e-stop?` badge never cleared after a resume. Every command for the
robot is routed to its host, so the proof that a lockout cleared - a command
the host accepted - was stamped on the host and never reached the robot's card.
A host's proof now covers the robots it holds, and `POST /api/mesh/safety/resume`
asks every live host for `state` (a read a locked peer refuses) once the resume
is heard, returns the peers that answered as `confirmed_clear` and pushes a
fresh snapshot. A peer that refuses or does not answer keeps `unknown`.
