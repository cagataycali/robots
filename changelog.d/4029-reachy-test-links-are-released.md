### Fixed: a Reachy driver a test connects releases its link thread with the test

`tests/drivers/test_reachy_driver.py` and `tests/drivers/test_reachy_oob_discovery.py`
connected a `ReachyDriver` in 33 tests and called `cleanup()` in 3, so each of the
others left the driver's loop thread - an idle `asyncio` `run_forever` with its
selector and self-pipe open - running on the xdist worker for the rest of its run:
114 of them measured on CI run `36095655976` (#4029). A shared fixture in
`tests/drivers/conftest.py` now records every driver at `_start_link`, the site that
creates the thread, cleans each up at teardown, and refuses a thread that started
during the test and is still alive afterwards, named by target - so a link started
by a path the fixture does not see fails the test that started it. Both modules opt
in once; no test spells its own `cleanup()` and the two that already did are
unchanged, since `cleanup` is idempotent. Test-only; no runtime behaviour changes.
