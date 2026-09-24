### Fixed: a distributed run reports its own truncation

`tests/session_truncation.py` sized a truncated session from `len(session.items)`
and `scripts/report_truncated_test_run.py` read the extent from `collected N
items`. Neither exists in a distributed session: a `pytest-xdist` controller
delegates collection to its workers, so it writes `2 workers [5 items]` instead
and its own item list stays empty. Measured on a 5-test tree stopped at the first
failure, `-n 2 -x` ran 3 of 5 and printed no section at all, and the script read
the same log as `unreadable` -- the one outcome that says nothing about whether
the suite ran. The forwarded `pytest_runtest_logfinish` events were counted
correctly all along; only the extent was missing.

The reporter now takes the collected count from the workers
(`pytest_xdist_node_collection_finished`, an optional hook, so the plugin still
registers where the wheel is absent), and the script reads the worker line as the
extent it is. That count is already net of deselection, and a distributed log
carries no collect-time `skipped` token, so the derived extent is bounded by what
the session selected -- the invariant `tests/` already asserted and now enforces.
