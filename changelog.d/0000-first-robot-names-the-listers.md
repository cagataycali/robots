### Docs: First robot names the listers that return a plain list

`docs/start/first-robot.md` said every call but `get_observation()` and
`cleanup()` returns the `status`/`content` envelope, yet its first code block
prints `list_robots()` and `robot_joint_names("so101")`, which return a plain
`list[str]` (as does `list_cameras()`), so `robot.list_robots()["status"]`
raised `TypeError`. The sentence now names the three listers, and the test
runs every call the page makes, not only the table rows.
