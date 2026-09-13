### Fixed: a robot that declares no category is grouped, named and ordered as `other`

`category` is optional in both the package registry and the user overlay, and
`register_robot` accepts `category=""`, so a robot that declares none is a
reachable state. `list_robots` reports its category as `""` - honest, because
that is what the entry says - and `list_robots_by_category` then read that key by
presence, `robot.get("category", "other")`. Its own input always supplies the
key, so the `"other"` fallback could never fire: such a robot came back under
`""`, a group with no name for a caller to switch on, and `format_robot_table`
rendered it as a blank Category cell. The repository's own table test asserted
the `"other"` grouping in its docstring while checking only that the row existed
at all, so the dead fallback survived.

The empty string was not the only unswitchable key. A declared category is
carried verbatim, so `category="  quadruped  "` opened a second group beside
`quadruped` holding the same robots' peers, and rendered just as blank as the
uncategorized row - one robot's padding split a category in two.

The grouping now reads the key by value and normalizes it:
`str(robot.get("category") or "").strip() or "other"`. A padded spelling joins
its own group instead of opening a blank-looking one beside it, and a robot that
declares nothing lands in a group with a name. `list_robots` is unchanged and
still reports each robot's `category` exactly as its entry declares it - the
normalization belongs to the grouping, not to the registry.

Two rendering consequences follow, and both are part of the same rule rather
than separate polish. `format_robot_table` now prints the group name in the
Category cell, so a robot appears under the group it was grouped in instead of
in a cell that names no group; and `"other"` is appended after the sorted custom
categories rather than sorted among them, because it is the absence of a category
and cannot outrank one somebody declared - sorted in, an uncategorized robot
rendered ahead of every custom category from `quadruped` onwards.

Measured over a registry of 77 robots with three registered for the measurement:
before, the returned keys included `''` and `'  quadruped  '` and no `'other'`,
and the uncategorized row rendered blank at index 74, ahead of the `quadruped`
row at 76. After, the keys carry `'other'`, `'  quadruped  '` is folded into
`quadruped`, and the uncategorized row renders as `other` last at index 76. The
group sizes still sum to `len(list_robots())`, which is pinned.
