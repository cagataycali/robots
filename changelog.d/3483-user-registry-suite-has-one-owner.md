### Fixed: the user-registry suite has one owner, and its alias pin is not inert

`tests/test_user_registry.py` and `tests/registry/test_user_registry.py` were
two copies of the same suite for `strands_robots.registry.user_registry`, and
they had drifted: 35 cells ran twice, 4 behaviours were pinned only at the root
path and 1 only under `tests/registry/`. The two copies ended up asserting
opposite contracts. The root copy pinned the shipped fail-closed behaviour - a
colliding alias is refused at `register_robot()` time - while the copy in the
canonical directory pinned a "never block a registration" fail-soft path by
patching `registry.robots.list_robots`, which `user_registry` no longer calls,
so its mock fired zero times and the cell passed vacuously.

The suite now lives only under `tests/registry/`, next to the module it
covers and under that directory's isolation conftest, with the alias-collision
and asset-dir pins carried over and the inert cell removed. Statement coverage
of `user_registry` is unchanged at 89%.
