### Fixed: the MuJoCo engine's own class name is now a typed export, not `Any`

`strands_robots.simulation` and `strands_robots.simulation.mujoco` each listed
names in `__all__` that the module never defined. They resolved at runtime
through a PEP 562 `__getattr__`, so nothing failed - but no static reader could
see them, and the affected name was the one that matters most: the class's own.

`simulation/__init__.py` maps three public spellings onto one class and mirrored
only the two aliases in its `if TYPE_CHECKING:` block, so `MuJoCoSimEngine`
revealed `Any` under mypy while `Simulation` and `MuJoCoSimulation` - the same
class object - revealed the concrete constructor signature. Every argument,
attribute and return value reached through the canonical spelling went
unchecked. `simulation/mujoco/__init__.py` had no `TYPE_CHECKING` block at all
and revealed a bare `type` for both of its exports.

Both are now imported under `TYPE_CHECKING`, which is static-only: no module is
imported at runtime that was not imported before, and all four spellings remain
the same object. This also closes CodeQL `py/undefined-export` alert 718, open
on `main` since 2026-07-09; the second module was never reported, because
assigning through `globals()[...]` reads as a definition to that analyzer.

`tests/test_all_exports_are_statically_defined.py` pins the contract across all
72 modules that declare a literal `__all__`, generalising an assertion that
previously covered `strands_robots/__init__.py` alone.
