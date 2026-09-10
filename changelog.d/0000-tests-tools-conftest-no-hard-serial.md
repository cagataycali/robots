### Fixed: `tests/tools` collects on an install without the `[lerobot]` extra

`tests/tools/conftest.py` imported `serial` at module scope. pyserial reaches
the tree only through `lerobot[feetech]`, so a venv without that extra failed
at conftest load (`ImportError while loading conftest`, exit 4) and collected
none of the 99 modules under `tests/tools`. The two fixtures that patch
`serial.Serial` now `pytest.importorskip("serial")` themselves; a grader pins
the conftest's module-scope imports to stdlib and pytest.
