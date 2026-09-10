### Fixed:

`[dev]` now declares `pyserial` and `msgpack`. `tests/tools/conftest.py` imports `serial` and the cosmos3 wire-format tests import `msgpack` at module level, so a `[dev]`-only environment (e.g. `uv sync --extra dev`) could not collect those directories and `scripts/check_whole_tree_graders.py` exited before grading anything. Both are small pure-Python packages; the lock changes only by those two entries.
