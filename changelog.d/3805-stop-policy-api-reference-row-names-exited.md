### Fixed: the API reference's `stop_policy` row names the fields the envelope really returns

`stop_policy` gained a bounded join and an `exited` verdict, and an empty
`robot_name` gained resolution to the only rollout in flight, but
`docs/api-reference.md` still described the answer as reporting `was_running`
alone. A caller reading the reference before their first call could not learn
that the answer tells them whether the robot is free yet. The row now names
`exited` and all three of its values, the join budget (graded against the
constant, not copied), and the empty-name resolution.
