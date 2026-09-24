### Changed: the test suite runs distributed, and a pull request's required check with it

The `test` script now carries `-n auto --dist loadfile`, so `hatch run test`
-- and with it `call-test-lint / Test and Lint`, the one required check --
spreads the suite over the runner's cores instead of running 58,000 tests in one
process. `loadfile` keeps every test of a file on one worker, which is what the
file-scoped fixtures and the module-level state this suite has assume. Coverage
is unaffected: it aggregates across workers, and the `--cov-fail-under=80` gate
is graded on the combined data (measured: 95%). The flags sit on that script and
not in `[tool.pytest.ini_options].addopts`, which every pytest invocation rooted
in the repository inherits: `tests_integ/` binds fixed ports, named containers,
one GPU and physical serial buses, so `hatch run test-integ` and a bare `pytest`
stay in one process, and a test pins the split.

Measured: the check's test step took 34:01 in one process on the runner; the
same suite with coverage on takes 12:49 and 15:57 in two runs at `-n 4` on four
pinned cores (the runner's core count), over 58,264 tests. Put `-n0` back on the
command line for a single test under `pdb`.

Two cells had been passing only because the suite was serial, and both now read
a quantity the scheduler cannot move rather than one it can:
`tests/test_mesh_pacing_ticker.py` graded each wait after a deliberate overrun,
where the first is legitimately short when the process is descheduled between
the overrun and the wait (6ms of a 20ms period under load) -- it now grades the
total, which is above two periods for any phase and near zero for the catch-up
burst the cell exists to catch. `tests/test_device_connect_stand_in_is_not_handed_back.py`
runs a nested pytest whose whole subject is the order two files import in, and
hands it this repository's config with `-c`; that nested run is now explicitly
`-n0`, because distributing those files is the ordering being measured.

Distributing the session also reaches the other half of the `robot_descriptions`
cache window that `strands_robots._description_cache` now owns for the package:
a test module imports a description through its own `importorskip`, which no
package seam sees, and every worker collects every file. The session wraps the
upstream `clone_to_cache` in *that* module's lock, so a collecting worker and a
package caller in one cache directory wait on one lock file instead of holding
one each. Measured on a cold cache with two workers over the two files that
import a description: 2 collection errors before, 31 passed after.
