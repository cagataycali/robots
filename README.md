# robot_mesh: a rate-limit slot must be reserved atomically on both gate paths

`capture.py` runs the same two-thread race in whichever tree it is placed in and
prints that tree, so the two dumps are attributable:

* `base-upstream-main.json` - measured at `upstream/main` (e94b214f)
* `branch.json` - measured on the fix branch
* `mutations.json` - the 6-row mutation table, both arms

`mutate.py` scopes every anchor to `robot_mesh`'s own AST line range, prints the
in-function vs in-file counts, and restores the source byte-identically.
`compose.py` asserts every drawn number against the dumps before saving.

The barrier holds both workers between the early `_rate_limit_check` and the
point a slot is taken, which is the window the fix closes.
