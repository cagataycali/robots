### Fixed: every `list_robots()` row lists the aliases that resolve to it

`list_robots()` rows carried seven fields and dropped the registry's `aliases`,
so `Robot("g1")` worked while no row mentioned `g1`, and the table footer
counted 139 aliases that no row showed. Each row now has an `aliases` list -
every spelling `list_aliases()` maps to that robot, sorted, empty when it
declares none - so the `unitree_g1` row reads `["g1", ...]`.
