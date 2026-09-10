# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""The session's ``MUJOCO_GL`` names a backend this platform's mujoco accepts.

``mujoco.rendering.classic.gl_context`` validates ``MUJOCO_GL`` once, at its
first import, and raises ``RuntimeError: invalid value for environment
variable MUJOCO_GL`` for a backend the platform does not have - ``egl`` on
macOS. Many test modules set the variable at import time with ``setdefault``,
so whichever module pytest collects first decided the value for the whole
session; on a Mac that was the first hard-coded ``"egl"``, and every later
render in the session failed. ``tests/conftest.py`` now sets a platform-valid
default before any test module is imported, which this guard pins by reading
the value the session actually ended up with.
"""

from __future__ import annotations

import os
import platform

#: Mirrors the table in ``mujoco/rendering/classic/gl_context.py``: the values
#: accepted everywhere, plus the platform's own backends.
_VALID_EVERYWHERE = {"enable", "enabled", "on", "true", "1", "glfw", ""}
_VALID_BY_SYSTEM = {"Linux": {"glx", "egl", "osmesa"}, "Windows": {"wgl"}, "Darwin": {"cgl"}}


def test_session_mujoco_gl_is_a_backend_this_platform_accepts() -> None:
    value = os.environ.get("MUJOCO_GL", "").lower().strip()
    accepted = _VALID_EVERYWHERE | _VALID_BY_SYSTEM.get(platform.system(), set())
    assert value in accepted, (
        f"MUJOCO_GL={value!r} is not a backend mujoco accepts on {platform.system()} "
        f"({sorted(accepted)}); a test module's import-time default leaked into the session"
    )
