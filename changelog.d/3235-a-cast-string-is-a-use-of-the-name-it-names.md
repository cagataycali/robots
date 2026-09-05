### Docs: a `cast("X", ...)` string is a use of `X`, and the CodeQL exemption for it has a stated boundary

CodeQL's `py/unused-import` reads bare `Name` loads, so a `TYPE_CHECKING` import
consumed only by a string forward reference inside `typing.cast` is reported as
dead. `AGENTS.md` now records the mechanism, the counterfactual that separates a
false positive from a real finding -- delete the import and lint the file: `F821`
at the cast means the import is the name's only binding, a clean `ruff check`
means the name is bound at runtime too and the alert is right -- and why the rule
is dismissed per instance rather than filtered.

The entry exists because the adjudication had been made three times and never
written down. Alert 599 was dismissed 32 minutes after it opened on 2026-07-02,
with the reasoning in a dismissal comment capped at 280 characters and living
outside the tree; alert 1138 then sat open on `main` for five days, and alert
1160 opened a review thread that held a merge for twelve hours under
`required_review_thread_resolution`.

`tests/test_cast_string_imports_are_the_names_only_binding.py` grades the
boundary rather than the prose. It derives the sites from the tree and refuses a
`TYPE_CHECKING` import of a name the module also binds at runtime -- the one
shape where this exemption would suppress a true finding, and the one direction
`ruff` and `mypy` cannot report, since the cast string resolves either way. No
production behaviour changes.
