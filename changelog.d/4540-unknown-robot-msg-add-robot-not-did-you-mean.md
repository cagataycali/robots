### Fixed: `_unknown_robot_msg` tells `add_robot` instead of hallucinating a typo

`Robot('so101')` + `run_policy('so100')` used to answer
`Robot 'so100' not found. Did you mean: so101? Available robots: ['so101'].`
when `so100` is a registered robot in its own right (both are in the registry).
The scene-local `close_match_hint` suggested the only loaded sibling as if the
caller mistyped, and the "available" listing implied so101 was the only robot
that exists. The only recovery is `add_robot('so100')`, which the message never
named.

`_unknown_robot_msg` now detects a registered name absent from the scene (via
`strands_robots.registry.get_robot`, which resolves canonical names *and*
aliases: `g1 -> unitree_g1`) and emits an actionable
`'so100' is a registered robot but is not loaded in this scene; add it first
with action='add_robot', name='so100'. Robots in the scene: ['so101'].` nudge
instead. The typo path (`so10`) and the empty-scene path are unchanged. When
the scene holds an alias of the requested robot (world has `g1`, caller asks
for `unitree_g1`), the message defers to the existing alias-aware
`close_match_hint` because the robot is already loaded under a different name.
