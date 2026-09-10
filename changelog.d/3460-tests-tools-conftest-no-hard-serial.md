### Fixed: `tests/tools` collects on an install without the `[lerobot]` extra

`tests/tools/conftest.py` imported `serial` at module scope. pyserial reaches
the tree only through `strands-robots[lerobot]` (which requests
`lerobot[feetech]`), so a `[dev]`-only install failed at conftest load
(`ImportError while loading conftest`, exit 4) and collected none of the 86
modules under `tests/tools`. That also masked
`tests/tools/test_tools_lazy_import.py`, which grades whether each tool imports
when its own extra is absent. The two fixtures that patch `serial.Serial` now
take pyserial at fixture time, and a grader pins every conftest's module-scope
imports to what a base install has.
