# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``serial_tool`` and ``pose_tool`` without pyserial are refused with the install line.

Both tools drive a Feetech bus through pyserial, and pyserial is declared by no
extra of this project on its own - it arrives only inside ``lerobot[feetech]``.
Measured on a fresh ``pip install strands-robots`` (main ``6abbd1a25``)::

    >>> from strands_robots import serial_tool
    UserWarning: serial_tool not available (missing dependencies): No module named 'serial'
    ImportError: cannot import name 'serial_tool' from 'strands_robots'

The two modules imported ``serial`` bare at their top, so the only remedy text a
customer had was the interpreter's. The Feetech *driver* next to them already
does this right (``FeetechBus.connect`` -> ``require_optional("serial",
pip_install="pyserial", ...)``). The tools now bind ``serial`` the same way at
import, so the package's lazy door (``strands_robots.__getattr__``) carries the
install line in its warning and a direct import raises it outright.

Cells use the shared :func:`blocked` helper (restores ``sys.modules`` and
``require_optional``'s memo) and :func:`reimport` (re-runs the module top level
and puts both of its bindings back), so nothing leaks into the session.
"""

from __future__ import annotations

import pytest

import strands_robots
from tests._blocked_module import blocked
from tests._module_reimport import reimport

TOOLS = ("serial_tool", "pose_tool")


@pytest.mark.parametrize("name", TOOLS)
def test_direct_import_without_pyserial_names_the_install_line(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Importing the tool module itself raises ``require_optional``'s refusal."""
    with blocked("serial"), pytest.raises(ImportError) as info:
        reimport(monkeypatch, f"strands_robots.tools.{name}")

    text = str(info.value)
    assert info.value.name == "serial", text
    assert "pip install pyserial" in text, text
    assert name in text, text  # the purpose names the door the caller came through


@pytest.mark.parametrize("name", TOOLS)
def test_package_door_without_pyserial_warns_with_the_install_line(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """``from strands_robots import <tool>`` surfaces the same line through the lazy loader's warning."""
    # The lazy loader memoises a successful load on the package; drop it so
    # this access goes through ``__getattr__`` again.
    monkeypatch.delattr(strands_robots, name, raising=False)
    with blocked("serial"):
        with pytest.raises(ImportError):
            reimport(monkeypatch, f"strands_robots.tools.{name}")
        with pytest.warns(UserWarning, match="pip install pyserial"), pytest.raises(AttributeError):
            getattr(strands_robots, name)
