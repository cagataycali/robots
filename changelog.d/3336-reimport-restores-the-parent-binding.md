### Tests: a re-imported module is put back on its parent package, not only in `sys.modules`

`importlib.import_module` binds a submodule in two places: the `sys.modules`
entry, and an attribute of the same name on its parent package.
`monkeypatch.delitem` restores the first. Two cells removed an entry so a
module's top level would run again -
`tests/simulation/test_policy_runner.py::test_policy_runner_import_does_not_pull_in_mujoco`
and
`tests/tools/g1/test_motion_switcher_decoder.py::test_module_can_be_imported_without_the_sdk` -
and put back one of the two. For the rest of the session the parent package
carried the fresh module while `sys.modules` held the original, and which of the
two a spelling reaches is not visible where it is written: `import a.b.c as m`,
`import a.b.c` and `from a.b import c` all resolve through the attribute, while
`sys.modules[...]`, `from a.b.c import f` and the live code's own globals resolve
through the entry.

Restoring one of the two is worse than restoring neither. With neither restored
both halves hold the fresh module and agree; with only the entry restored a cell
that binds the module inside its body and patches an attribute on it patches the
copy production does not read, so the stub is inert and the cell grades nothing
while still passing.

That was live. `tests/tools/test_run_policy.py` binds the runner in-cell to
install a `PolicyRunner` that raises, and its two cells pin that
`strands_robots.tools.run_policy._finalize_episode` tolerates a construction
error and a save error. With both `except Exception` guards deleted from that
function, the two cells failed on their own and passed after the split was left
behind - the ordering the full suite collects, `tests/simulation` before
`tests/tools`.

`tests/_module_reimport.reimport` records both bindings before importing, and
both cells ask it instead of spelling the pair out. `tests/test_sys_modules_removal_leaves_no_orphan.py`
grades the rule tree-wide beside the orphan rule it already carried, reading the
key through a local binding because that is how the graded cells spell it, and
leaving alone the two remaining remove-and-import cells whose shape cannot split:
one drives an import that raises, which binds nothing, and one never restores the
entry.
