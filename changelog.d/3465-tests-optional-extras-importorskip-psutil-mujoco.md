### Fixed: 48 test modules skip, rather than error, on a venv without `[sim-mujoco]` or `[lerobot]`

`mujoco` is declared only by `[sim-mujoco]` and `psutil` only by `[lerobot]`, yet
48 test modules reached one at module scope with no `pytest.importorskip` above
the import: 19 under `tests/simulation` imported `mujoco` directly (18 siblings
in the same directory already had the guard), and 29 reached `psutil` through
`strands_robots.tools.lerobot_train`, `lerobot_teleoperate` or `_process_stop` --
one of them behind an `importorskip("pyarrow")` that hid the hole. With the
dependency absent each was a collection ERROR; all 48 now collect as SKIPPED,
and the same 1,939 tests still collect where the extra is installed. A grader
parametrized over the two dependencies pins the rule for every module under
`tests/`, deriving the `psutil` carriers from the package's own module-scope
imports so a new `import psutil` widens the check with no edit to the test.
