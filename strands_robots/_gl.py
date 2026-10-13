"""Pick MuJoCo's GL backend before the first ``import mujoco`` (it reads MUJOCO_GL once).

Linux without a display → ``egl`` if libEGL loads, else ``osmesa``; macOS/Windows use the native
context MuJoCo picks itself. An explicit ``MUJOCO_GL`` is always respected.
"""

from __future__ import annotations

import ctypes
import os
import platform


def _loads(lib: str) -> bool:
    try:
        ctypes.CDLL(lib)
        return True
    except OSError:
        return False


def ensure_gl() -> str:
    """Set ``MUJOCO_GL`` for headless rendering when it is unset. Returns the value in effect."""
    cur = os.environ.get("MUJOCO_GL", "").strip().lower()
    if cur:
        return cur
    if platform.system() == "Linux" and not os.environ.get("DISPLAY"):
        for value, lib in (("egl", "libEGL.so.1"), ("osmesa", "libOSMesa.so")):
            if _loads(lib):
                os.environ["MUJOCO_GL"] = value
                return value
    return cur
