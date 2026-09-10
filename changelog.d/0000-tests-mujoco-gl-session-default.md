### Fixed: the test session picks MuJoCo's GL backend once, on a value the platform accepts

29 test modules set `MUJOCO_GL` at import time with `setdefault`, 18 of them
hard-coded to `egl`, which mujoco refuses on macOS. pytest imports every
module during collection, so on a Mac the first of those decided the value for
the whole session and every later render failed - visible as the
order-dependent failure of `test_collection_loop_resets_between_episodes.py`
under `pytest tests -k ...` (8/8 alone, "Missing features:
observation.images.default" in a combined run). `tests/conftest.py` now sets a
platform-valid default before any test module is imported, so the per-module
lines are no-ops; a guard reads the value the session ended up with and checks
it against mujoco's own table for the platform.
