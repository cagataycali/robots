### Docs: a `Pinned by` coordinate in `AGENTS.md` names a test that exists

`AGENTS.md` cited `tests/test_examples_mujoco_gl.py::test_no_module_scope_windowed_gl_default`,
and that test module's own docstring named the same function in a `:func:` role, after the
test had been renamed `test_no_module_scope_platform_bound_gl_default` when its rule was
widened from the windowed pair to every platform-bound backend. A reader who copied the
coordinate into `pytest` got `not found`, so the rule read as enforced while the pin it
named could not be run. Both citations now spell the defined name, and
`tests/test_agents_md_pinned_by_citations_resolve.py` grades every `path::Name` coordinate
in `AGENTS.md` against the cited file's AST - derived from the document, no import, so a
pin whose module needs an optional extra is still graded.
