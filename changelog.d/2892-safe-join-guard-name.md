### Docs: the traversal guard is named by the spelling it is defined with

`strands_robots/utils.py` defines the traversal guard as `safe_join`, and its 13
call sites across `assets/manager.py`, `assets/download.py`,
`simulation/task_objects/catalog.py` and `tools/harness_memory.py` spell it that
way. Two pieces of prose spelled it `_safe_join`: the `.. warning:: Security`
block on `register_robot`, which tells a future implementer to "validate all
paths with" it, and the registry-conventions section of `AGENTS.md`, which
attributed it to the right file under the wrong name.

A reader who grepped the documented spelling therefore found no implementation,
because the only `_safe_join` in the tree was a synthetic stub inside a test
fixture string. That conclusion was reached and written into a repository audit,
which reported the guard as a phantom with no implementation anywhere in
`strands_robots/`, while it was present and tested the whole time. A security
warning naming a symbol nobody can find reads as though the remediation were
still unwritten.

The six references now name `safe_join`, and the security warning points at it
through a resolvable cross-reference. No behaviour changes. A new sweep grades
every public callable `utils.py` defines and derives its exemptions from the
tree, so an underscored spelling that resolves to nothing is refused while a real
private wrapper such as `_coerce_rgba` is not.
