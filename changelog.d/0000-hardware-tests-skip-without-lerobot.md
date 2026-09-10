### Fixed
- `tests/test_hardware_camera_rollback.py` and `tests/test_hardware_cleanup_disconnects.py` skip when the `lerobot` extra is absent instead of failing collection, the way the other 145 lerobot-dependent tests already do.
