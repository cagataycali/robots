### Tests: the two hardware test modules skip when lerobot is absent

`tests/test_hardware_camera_rollback.py` and `tests/test_hardware_cleanup_disconnects.py` imported `lerobot.utils.errors` at module level, so a core-only install aborted collection with two errors instead of skipping. They now gate on `pytest.importorskip("lerobot")` the way the other lerobot-dependent modules do.
