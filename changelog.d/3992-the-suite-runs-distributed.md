### Changed: the test suite runs distributed, and a pull request's required check with it

`[tool.pytest.ini_options].addopts` now carries `-n auto --dist loadfile`, so
`hatch run test` -- and with it `call-test-lint / Test and Lint`, the one
required check -- spreads the suite over the runner's cores instead of running
58,000 tests in one process. `loadfile` keeps every test of a file on one
worker, which is what the file-scoped fixtures and the module-level state this
suite has assume. Coverage is unaffected: it aggregates across workers, and the
`--cov-fail-under=80` gate is graded on the combined data (measured: 95%).

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
