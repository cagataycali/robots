### Changed: an undocumented public surface is reported everywhere, by the linter

The rule "a public module, class, method or function has a docstring" was
asserted by fourteen copies of one AST walk under `tests/`, one per package,
each rooted at its own `<pkg>.__file__`. Fourteen copies reached fourteen of the
package's forty-one directories, and all twenty-two undocumented surfaces in the
tree sat in directories none of them walked: `dashboard/` (eighteen, including
the WebAuthn ceremony verbs `verify_token` / `begin_authentication` /
`finish_authentication` / `finish_registration` / `status`), `rtps/idl/` (two)
and `tools/g1/` (three - `list_services`, `list_operations` and
`describe_operation`, the discovery verbs an agent calls precisely because it
does not know the surface). The 37 cells across those fourteen files passed on
that tree, because a per-package guard reports a clean sweep over whichever
packages it happens to name.

`ruff` now selects the pydocstyle PRESENCE codes -- `D100`, `D101`, `D102`,
`D103`, `D106` -- over the whole package inside the merge-blocking lint gate, and
the twenty-two surfaces are documented. The FORMAT codes (`D205` summary layout,
`D401` imperative mood, `D413` section spacing, ...) are deliberately not
selected: 2,051 findings on this tree, none of them a missing docstring.
Completeness of a docstring that exists is unchanged and still graded against
the blocks it already has, by `tests/test_args_docstring_completeness.py` and
`tests/test_raises_docstring_completeness.py`.

The fourteen guards are removed. Their per-package `_EXPECTED_CLASSES` /
`_EXPECTED_FUNCTIONS` rosters were non-vacuity guards for their own scan by
their own docstrings ("the scan actually found the classes and functions it
protects"); a linter handed a named directory has no such failure mode, and a
dropped or renamed public symbol is owned by the griffe breaking-change check.
