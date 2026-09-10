### Fixed:

`[dev]` now declares `pyserial` and `msgpack`, which `tests/tools/conftest.py` and the cosmos3 wire-format tests import at module level, so a `[dev]`-only environment no longer fails at the conftest before collecting anything. The suite and `scripts/check_whole_tree_graders.py` still need `.[all,dev]` (tests/tools imports the `[lerobot]` extra directly); `AGENTS.md` now says so next to the install command.
