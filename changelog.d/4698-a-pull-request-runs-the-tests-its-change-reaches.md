### Changed: a pull request's required check runs the tests its change reaches

`hatch run test` now goes through `scripts/select_tests.py`. On a
`pull_request` event with no test path named, it runs the test files that name
a changed module (by import, by a dotted string, through a test helper, or
through a class that inherits from it), the whole-tree graders, the tests under
a changed `conftest.py`, and the tests that spell a path into a changed
`docs/`, `examples/` or `changelog.d/` file, with `--no-cov`. A change to
`pyproject.toml`, `uv.lock`, `scripts/`, `.github/` or `tests/conftest.py`, and
every push to `main`, still runs the whole suite with the coverage floor.
`python scripts/select_tests.py --list` prints what a branch would run.
