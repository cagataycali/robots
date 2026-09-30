### Fixed: a stop acknowledged by a locked peer no longer paints its dashboard card clear

A peer whose e-stop lockout is engaged still answers `status`, `resume`,
`stop` and `ping`; the dashboard treated an acknowledged `stop` (and `ping`)
as proof the lockout had cleared, so STOP ALL against an already locked fleet
annotated every still-locked robot `state="clear"` on the next snapshot. The
peer and the dashboard now read one set,
`strands_robots.mesh.security.LOCKOUT_ADMITTED_ACTIONS`, and answering any
action in it proves nothing; only a command the lockout would have refused
clears a card.
