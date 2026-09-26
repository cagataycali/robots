### Changed: a passed test's `current` symlink is removed with its `tmp_path`

`tmp_path_retention_policy = "failed"` removes a passed test's directory at its
own teardown, but pytest writes a second entry into the base temp for every
`tmp_path` - a `<name>current` symlink to it, one per distinct 30-character
test-name prefix - and removes those only at session end. `make_numbered_dir`
lists the whole base temp to name the next directory, so under the policy a
worker's setup cost still grew with the distinct test names it had run:
~25,600 across the suite, ~13,000 per worker. Measured on the two-vCPU runner
with 20,000 trivial tests of distinct names under one autouse `tmp_path`
fixture, `-n 2 --dist loadfile`: 101-108 s, against 42.7 s for the 250-name
suite the policy was measured on, and a mid-run sample of each worker's base
temp held 7,000+ symlinks and no directory. A `pytest_runtest_teardown`
wrapper in `tests/conftest.py` now unlinks the symlink once the directory is
gone - a failed test keeps both, as before - which took the same 20,000 to
44-45 s (34 s with no `tmp_path` at all). Pinned by
`tests/test_tmp_path_of_a_passed_test_is_not_kept.py`. Towards #3869.
