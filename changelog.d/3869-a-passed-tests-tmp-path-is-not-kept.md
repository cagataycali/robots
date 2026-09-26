### Changed: a passed test's `tmp_path` is removed at its own teardown

`[tool.pytest.ini_options]` now sets `tmp_path_retention_policy = "failed"`.
Every test in the suite creates a `tmp_path` - the autouse fixtures in
`tests/conftest.py` that redirect the dashboard credential store and the
session store request it - and pytest's default `"all"` kept each of the
~58,000 directories until the session ended, while `make_numbered_dir` lists
the whole base temp to name the next one. So a worker's N-th test paid a scan
of the N-1 directories before it, ~10-20 ms of setup per test late in a run,
which no per-cell tail shows. Measured on the two-vCPU runner with trivial
tests under one autouse `tmp_path` fixture, `-n 2 --dist loadfile`: 10,000
tests 34.3 s -> 21.1 s and 20,000 tests 96.7 s -> 42.7 s, against 12.9 s with
no `tmp_path` at all. The same directory is the cwd of the nested pytest runs
in `tests/test_session_truncation_is_reported.py`, which adopt it as their
rootdir and collect it: 4.5 s per run against 24,000 sibling entries, 0.27 s
without them. A failed test's directory is kept, which is the one anyone
inspects; pytest removes its `<name>current` symlinks only at session end, so
within a run the base temp still grows with the number of distinct test names
(~25,600 against 58,000 tests). Pinned by
`tests/test_tmp_path_of_a_passed_test_is_not_kept.py`. Towards #3869.
