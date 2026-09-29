### Fixed: the unit suite never boots a real Isaac Sim

On a machine with Isaac Sim installed, unit tests that reached the Isaac backend's lazy import booted a real Kit app inside the pytest worker, hanging on the EULA prompt or failing at random. `tests/conftest.py` now blocks a real `isaacsim` import for unit tests, as on a machine without Isaac.
