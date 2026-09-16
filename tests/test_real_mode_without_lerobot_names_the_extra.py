# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``Robot(..., mode="real")`` without lerobot is refused with the extra that supplies it.

Measured on ``main`` with lerobot made unimportable, for every kwarg shape the
README's real-arm quickstart can take (with a port, with cameras, with neither)::

    >>> Robot("so101", mode="real", port="/dev/…")
    ModuleNotFoundError  (hardware_robot.py:1140, name='lerobot.robots')

The interpreter's message was the whole answer: no extra, no ``pip`` line, and
an ``ImportError.name`` naming a submodule rather than the distribution to
install. Every other door a core install can walk through already refuses with
a remedy -- ``Robot("so101")`` names ``[sim-mujoco]``, ``python -m
strands_robots dashboard`` names ``[dashboard]``, the Feetech bus names
``pyserial`` -- because each goes through
:func:`strands_robots.utils.require_optional` (AGENTS.md convention 7).
``_initialize_robot`` imported lerobot bare, and a comment in ``__init__`` even
recorded that a "lerobot ImportError" would be raised there first; it just
never said which package to install.

The refusal is scoped to the lerobot driver, not to real mode at large, and the
second cell is what holds it to that: ``driver="strands"`` never reaches
``_initialize_robot``, so all 25 robots with a native driver registered
(``so101`` and ``panda`` among them) are still built with lerobot absent. A
purpose line reading "required for real-mode robots" would be false for each of
them, which is why it names the driver instead.

The ``name`` assertions are the machine-readable half:
``tests/test_absent_dependency_reports_name_the_module.py`` explains why a
reader must not have to parse the prose to learn which module was absent.
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_robots import Robot
from strands_robots.drivers.base import HardwareDriver
from tests._blocked_module import blocked

#: The kwarg shapes a real-mode call arrives in. All three reached the bare
#: import on main, so all three are graded here.
_QUICKSTART_CALLS: list[tuple[str, dict[str, Any]]] = [
    ("port only", {"port": "/dev/null"}),
    ("port and cameras", {"port": "/dev/null", "cameras": {"front": {"type": "opencv", "index_or_path": 0}}}),
    ("no kwargs", {}),
]

#: The driver values that route through lerobot: the explicit one and the
#: default, which resolves to it for a robot whose registry names no other.
_LEROBOT_DRIVERS = ["auto", "lerobot"]


@pytest.mark.parametrize(("shape", "kwargs"), _QUICKSTART_CALLS, ids=[c[0] for c in _QUICKSTART_CALLS])
def test_real_mode_without_lerobot_names_the_lerobot_extra(shape: str, kwargs: dict[str, Any]) -> None:
    """The refusal names ``lerobot``, the ``[lerobot]`` extra and the ``pip`` line."""
    with blocked("lerobot"), pytest.raises(ImportError) as info:
        Robot("so101", mode="real", **kwargs)

    exc = info.value
    text = str(exc)
    assert exc.name == "lerobot", text
    assert "pip install 'strands-robots[lerobot]'" in text, text
    assert "pip install lerobot" in text, text
    # The purpose says which door the caller came through, so the same message
    # read out of a log later is still attributable.
    assert 'mode="real"' in text, text
    assert "lerobot driver" in text, text


@pytest.mark.parametrize("driver", _LEROBOT_DRIVERS)
def test_the_default_driver_and_the_lerobot_driver_both_refuse(driver: str) -> None:
    """Both routes into ``_initialize_robot`` refuse alike, so neither is a silent one."""
    with blocked("lerobot"), pytest.raises(ImportError) as info:
        Robot("so101", mode="real", driver=driver, port="/dev/null")

    assert info.value.name == "lerobot", str(info.value)


@pytest.mark.parametrize("name", ["so101", "panda", "ur5e"])
def test_a_native_driver_real_robot_is_built_without_lerobot(name: str) -> None:
    """``driver="strands"`` needs no lerobot, which is what scopes the refusal above.

    The guard sits in ``_initialize_robot``, on the lerobot branch of the
    factory only. Were it moved up into ``Robot()`` -- or were the purpose line
    widened to real mode at large -- these robots would be told to install a
    package their driver never imports.
    """
    with blocked("lerobot"):
        built = Robot(name, mode="real", driver="strands", port="/dev/null")

    assert isinstance(built, HardwareDriver)
