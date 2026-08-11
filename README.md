# quantile-norm probe coverage - measurement scripts

`census.py` / `matrix.py` are the coverage-census views that located the gap
(`census.py` reads a `--cov-report=json` dump; `matrix.py` is the guard-refusal
matrix view, which was empty on this base).

`capture.py` re-measures everything the figure shows into one JSON: the
eight-probe sibling matrix from the pristine full-suite coverage dump, the
target lines before/after the new tests, the four-row consequence table, and
the mutation table over both arms (this PR's tests vs upstream's copy of the
same file). `compose.py` renders it and asserts every rendered number against
that JSON before saving.

Run from a checkout root with `PYTHONPATH=<root>` and `GITHUB_RUN_ID` set to
whatever suffix the cached `/tmp/cov-*.json` dumps use.
