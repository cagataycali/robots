### Fixed: a MoveIt2 plan that commands nothing is a planning failure, not a successful plan

The sidecar's `planner_returned_empty` guard only inspected the plan object, so a
truthy plan that serialised to zero waypoints - or to waypoints carrying no
`positions` - came back as `success=True, status="ok"`. The client then unpacked
that into zero actions, or into one empty action dict per waypoint: a plan the
caller counts as having waypoints while it moves no joint, which the downstream
"empty action chunk" guards cannot see because the list is not empty. An
unusable row was also dropped silently, so the path executed was the plan minus
a waypoint the collision-aware planner had routed around.

The sidecar now reports such a plan as `planner_returned_empty:<detail>` - same
status kind, so a client already matching it needs no change - and the client
refuses a `success=True` reply that carries no waypoint or a waypoint with no
joint position. `get_actions` never returns an empty list or an action dict that
commands nothing.
