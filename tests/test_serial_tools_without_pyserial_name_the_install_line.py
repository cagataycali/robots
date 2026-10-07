# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every door that needs pyserial is refused with the ``[serial]`` extra's install line.

Four doors talk to a serial bus through pyserial: the ``serial_tool`` and
``pose_tool`` agent tools at import, and the native Feetech and Dynamixel buses
at ``connect()``. ``[serial]`` declares pyserial with the project's bound, so the
refusal names that extra first and the bare distribution second, the way every
other native driver's refusal names its own extra. ``ImportError.name`` stays
``"serial"`` so a caller can tell an absent extra from a broken package path
(AGENTS.md convention 7).

Cells use the shared :func:`blocked` helper (restores ``sys.modules`` and
``require_optional``'s memo) and :func:`reimport` (re-runs the module top level
and puts both of its bindings back), so nothing leaks into the session.
"""

from __future__ import annotations

import tomllib
from collections.abc import Callable
from pathlib import Path

import pytest

import strands_robots
from strands_robots.drivers.dynamixel.bus import DynamixelBus
from strands_robots.drivers.feetech.bus import FeetechBus
from tests._blocked_module import blocked
from tests._module_reimport import reimport

TOOLS = ("serial_tool", "pose_tool")
INSTALL_LINES = "Install with:\n  pip install 'strands-robots[serial]'\n  pip install pyserial"


def _import_tool(name: str) -> Callable[[pytest.MonkeyPatch], object]:
    return lambda monkeypatch: reimport(monkeypatch, f"strands_robots.tools.{name}")


def _connect(bus: type[FeetechBus]) -> Callable[[pytest.MonkeyPatch], object]:
    return lambda monkeypatch: bus(port="/dev/ttyACM0").connect()


@pytest.mark.parametrize(
    ("door", "purpose"),
    [
        (_import_tool("serial_tool"), "serial_tool"),
        (_import_tool("pose_tool"), "pose_tool"),
        (_connect(FeetechBus), FeetechBus.PURPOSE),
        (_connect(DynamixelBus), DynamixelBus.PURPOSE),
    ],
    ids=["serial_tool", "pose_tool", "FeetechBus.connect", "DynamixelBus.connect"],
)
def test_without_pyserial_the_refusal_names_the_serial_extra(
    monkeypatch: pytest.MonkeyPatch, door: Callable[[pytest.MonkeyPatch], object], purpose: str
) -> None:
    with blocked("serial"), pytest.raises(ImportError) as info:
        door(monkeypatch)

    text = str(info.value)
    assert info.value.name == "serial", text
    assert text.endswith(INSTALL_LINES), text
    assert purpose in text, text  # the purpose names the door the caller came through


def test_the_serial_extra_declares_pyserial_and_dashboard_reaches_it_through_the_extra() -> None:
    extras = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]
    assert extras["serial"] == ["pyserial>=3.5,<4.0"]
    assert "strands-robots[serial]" in extras["dashboard"]
    assert not [r for r in extras["dashboard"] if r.startswith("pyserial")], "a second pyserial bound can drift"


@pytest.mark.parametrize("name", TOOLS)
def test_package_door_without_pyserial_warns_with_the_install_line(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """``from strands_robots import <tool>`` surfaces the same line through the lazy loader's warning."""
    # The lazy loader memoises a successful load on the package; drop it so
    # this access goes through ``__getattr__`` again.
    monkeypatch.delattr(strands_robots, name, raising=False)
    with blocked("serial"):
        with pytest.raises(ImportError):
            reimport(monkeypatch, f"strands_robots.tools.{name}")
        with pytest.warns(UserWarning, match=r"strands-robots\[serial\]"), pytest.raises(AttributeError):
            getattr(strands_robots, name)
