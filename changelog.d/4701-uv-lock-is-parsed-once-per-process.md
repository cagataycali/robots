### Tests: the packaging rules read `uv.lock` once per process, not once per cell

Seven packaging test files parsed the 1.5 MB lock on their own, several once
per cell: 70 parses in one process. They now share
`tests.uv_lock_closure.uv_lock()`, which parses it once, and
`tests/test_package_sources_are_parsed_once_per_process.py` refuses a fresh
`tomllib` read of the lock anywhere else under `tests/`. The thirteen files take
42 s instead of 58 s on one core.
