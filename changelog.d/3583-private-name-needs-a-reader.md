### Changed: a private name must have a reader

Two private module-level names were defined and never read anywhere in the
tree. `_VALID_MODES` in `strands_robots.robot` named the three mode spellings
while every site that decides or reports a mode - `_auto_detect_mode`'s
membership test, its warning and the factory's `ValueError` - spelled them
inline, so the constant documented an owner that did not exist.
`_PEERS_VERSION` in `strands_robots.mesh.session` was a registry version
counter incremented under `_PEERS_LOCK` at every insert, eviction, prune and
clear, and consulted by nothing: four write sites and three `global`
declarations maintaining a change signal for a cache that was never built.

Both are removed. `tests/test_no_private_name_is_defined_and_never_read.py`
pins the invariant for module-level private names and private methods across
`strands_robots/`, the module-scope half of what
`test_no_private_state_is_written_and_never_read.py` holds for instance
attributes. A reader is a load of the identifier, an attribute or import of
it, a string literal containing it, or any mention under `tests/`,
`tests_integ/`, `examples/` or `scripts/` - an example reaches a backend's
private helper through `getattr`, and it ships in the same tree. The grader
excludes its own file from that read, so naming an offender in its docstring
does not clear it.
