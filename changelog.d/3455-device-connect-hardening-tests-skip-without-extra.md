### Fixed: the Device Connect hardening tests skip without the extra instead of erroring

`tests/test_device_connect_hardening.py` reaches for the real
`device_connect_edge` package from a fixture helper, so in an environment
without the `[device-connect]` extra every one of its tests failed at setup with
`ModuleNotFoundError` -- 71 errors that say nothing about the tree under test.
A module-level `pytest.importorskip` makes the whole module one honest skip.
