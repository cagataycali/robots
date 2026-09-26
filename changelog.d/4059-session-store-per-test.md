### Changed: a test's session store is its own, not the store of the machine running the suite

`strands_robots.tools._process_stop.SESSION_DIR` is
`Path.cwd() / ".strands_robots/.sessions"`, bound as the module imports, so a
suite run from a project directory resolves it to that project's real session
store. That store is one document: `SessionManager` loads every record to
change one and writes them all back, and `remove_session` deletes out of it --
so a `stop` or `list` cell reaching for a record it wrote takes the operator's
live teleoperation and training records with it, under a green report, because
nothing in those cells asks where the store was. Measured with the new fixture
removed, a run of `tests/tools` wrote
`.strands_robots/.sessions/active_sessions.json` into the working tree.

Twenty-four test modules redirected it for themselves at twenty-six sites --
eleven in an autouse fixture whose body was the same three lines under five
different docstrings, the rest inline in a helper or a cell, under four fixture
names and three different paths. One autouse fixture in `tests/conftest.py`
now names `tmp_path / ".sessions"`, the path twenty of them chose, so a
module's own helper that seeds or reads the store needs no change. It names the
directory without creating it -- `session_log_path` and `store_sessions` make
it when something actually writes -- so a cell that lists its own `tmp_path`
still finds it empty, and the helpers that write a file into the store directly
make it themselves.

`tests/test_the_session_store_a_test_reaches_is_its_own.py` declares no fixture
of its own: the guarantee is that the redirect is a property of the session
rather than of twenty-four authors remembering to. Test-only change; no
behaviour of the package changed.
