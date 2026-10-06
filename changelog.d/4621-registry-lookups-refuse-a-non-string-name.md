### Fixed: registry lookups refuse a name that is not a string instead of raising `AttributeError`

`get_robot(None)`, `resolve_name(42)`, `has_sim(b"so100")` and the other
registry reads escaped as `AttributeError: 'NoneType' object has no attribute
'lower'` or a `TypeError` from `str.replace`, because each folds its argument
through `normalize_robot_name` first. That fold now raises the `ValueError`
`Robot(name)` already gives: `Invalid robot name 42 (int): a robot name is a
string. Pass a registered name (see list_robots()).`
