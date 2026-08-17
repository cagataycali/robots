### Fixed

- **tests**: an optional-dependency guard that bound a name inside a `try` whose handler calls `pytest.skip` and read it after the block now binds it on every path to its use, clearing five `py/uninitialized-local-variable` alerts. Missing modules go through `pytest.importorskip`; a missing attribute keeps its own skip so an upstream rename stays a skip rather than becoming an `AttributeError`; a value that has to be built comes back from a `*_or_skip` helper. A new check sweeps `tests/` and `tests_integ/` for the shape.
