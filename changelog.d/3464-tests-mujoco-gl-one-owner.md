### Changed: `tests/conftest.py` is the one place the test session picks `MUJOCO_GL`

The 29 test modules that each set `MUJOCO_GL` at import time with `setdefault`
drop the line (and the `os`/`sys` imports that only served it): the session
default in `tests/conftest.py` runs before any of them is imported, so every
one was already a no-op. The scanner's floor in `test_examples_mujoco_gl.py`
drops to 10 for the defaults that remain in `examples/` and `tests_integ/`,
which have no shared conftest.
