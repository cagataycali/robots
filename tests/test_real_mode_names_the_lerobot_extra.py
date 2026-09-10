"""``Robot(..., mode="real")`` without lerobot must name the extra to install.

Measured on a checkout synced without ``[lerobot]``::

    >>> Robot("so101", mode="real")
    ModuleNotFoundError: No module named 'lerobot'      # hardware_robot.py:920

-- the first hardware line a customer types, answered with a bare interpreter
error: no extra named, no install line, while every other optional surface in
the tree answers through ``require_optional``.
"""

from __future__ import annotations

import builtins
import sys

import pytest

from strands_robots import utils as utils_mod
from strands_robots.hardware_robot import Robot as HardwareRobot


def test_missing_lerobot_names_the_extra_and_the_pip_line(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRANDS_MESH", "false")
    monkeypatch.setattr(utils_mod, "_lazy_modules", {}, raising=False)
    for name in [m for m in sys.modules if m == "lerobot" or m.startswith("lerobot.")]:
        monkeypatch.delitem(sys.modules, name)
    real_import = builtins.__import__

    def _no_lerobot(name, *args, **kwargs):
        if name == "lerobot" or name.startswith("lerobot."):
            raise ModuleNotFoundError(f"No module named '{name}'", name=name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_lerobot)

    with pytest.raises(ImportError) as info:
        HardwareRobot("so101", "so101", port="/dev/null")

    text = str(info.value)
    assert info.value.name == "lerobot"
    assert "pip install 'strands-robots[lerobot]'" in text, text
    assert 'mode="real"' in text, text
