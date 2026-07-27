"""A leader arm is a teleoperator, not a robot.

``so101_leader`` was an alias of the ``so101`` registry entry, whose
``hardware.lerobot_type`` is ``so101_follower``. So
``Robot("so101_leader", mode="real", port=<leader port>)`` built an
SO101Follower driver on the leader's motor bus - torque-enabling the arm a
human is holding and turning it into a rigid position servo. These tests pin
the refusal and the registry invariant behind it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import strands_robots.hardware_robot as hardware_robot
import strands_robots.robot as robot_module
from strands_robots import Robot
from strands_robots.registry import get_hardware_type, get_robot, resolve_name

REGISTRY_PATH = Path(__file__).parent.parent / "strands_robots" / "registry" / "robots.json"


@pytest.fixture(scope="module")
def registry() -> dict[str, Any]:
    """Load the shipped robot registry once per module."""
    with open(REGISTRY_PATH) as f:
        data = json.load(f)
    entries: dict[str, Any] = data.get("robots", data)
    return entries


def test_no_leader_name_resolves_to_a_follower_entry(registry: dict[str, Any]) -> None:
    """Registry invariant: a ``*_leader`` name never names a follower robot.

    Generalises the ``so101_leader`` regression to the whole file - a leader
    name may only appear as a key or alias of an entry that is itself that
    leader device (``hardware.lerobot_type`` matching), never of the follower
    it drives.
    """
    offenders = []
    for name, info in registry.items():
        lerobot_type = (info.get("hardware") or {}).get("lerobot_type")
        for candidate in (name, *info.get("aliases", [])):
            if candidate.endswith("_leader") and candidate != lerobot_type:
                offenders.append(f"{candidate!r} -> entry {name!r} (lerobot_type={lerobot_type!r})")
    assert not offenders, "leader names resolving to a non-leader entry: " + "; ".join(offenders)


def test_so101_leader_does_not_resolve_to_the_follower() -> None:
    """The leader name must not carry the follower's hardware type."""
    assert resolve_name("so101_leader") == "so101_leader"
    assert get_robot("so101_leader") is None
    assert get_hardware_type("so101_leader") is None


def test_real_mode_refuses_a_leader_before_touching_hardware(monkeypatch: pytest.MonkeyPatch) -> None:
    """The refusal lands before any driver is constructed on the leader's port."""
    constructed: list[dict[str, Any]] = []

    class _Recorder:
        def __init__(self, **kwargs: Any) -> None:
            constructed.append(kwargs)

    monkeypatch.setattr(hardware_robot, "Robot", _Recorder)

    with pytest.raises(ValueError, match="teleoperator"):
        Robot("so101_leader", mode="real", port="/dev/ttyACM1")

    assert constructed == []


def test_the_refusal_names_the_teleoperator_route(monkeypatch: pytest.MonkeyPatch) -> None:
    """The message points at ``Teleoperator``/``attach_teleop``, not the follower.

    Answering with the follower name would invite the caller to retry
    ``Robot("so101", mode="real", port=<leader port>)`` - the same hazard by
    another spelling.
    """
    monkeypatch.setattr(hardware_robot, "Robot", lambda **kwargs: pytest.fail("built a driver for a leader"))

    with pytest.raises(ValueError) as excinfo:
        Robot("so101_leader", mode="real", port="/dev/ttyACM1")

    message = str(excinfo.value)
    assert "Teleoperator('so101_leader', port=...)" in message
    assert "attach_teleop('so101_leader', port=...)" in message
    assert "so101_follower" not in message


@pytest.mark.parametrize("mode", ["sim", "real", "auto"])
@pytest.mark.parametrize("name", ["so101_leader", "SO101-Leader", "so100_leader", "koch_leader"])
def test_a_leader_is_refused_in_every_mode(mode: str, name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """No mode accepts a leader - sim would silently hand back the follower."""
    monkeypatch.setattr(hardware_robot, "Robot", lambda **kwargs: pytest.fail("built a driver for a leader"))

    with pytest.raises(ValueError, match="is a teleoperator .leader. device, not a robot"):
        Robot(name, mode=mode, port="/dev/ttyACM1")


def test_follower_names_still_resolve() -> None:
    """The follower aliases this fix did not touch keep resolving."""
    assert resolve_name("so101_follower") == "so101"
    assert resolve_name("so101_dualcam") == "so101"
    assert get_hardware_type("so101") == "so101_follower"


def test_an_unknown_non_leader_name_keeps_the_registry_listing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The leader branch must not hijack the generic unknown-robot error."""
    monkeypatch.setattr(robot_module, "is_discoverable", lambda name: False)

    with pytest.raises(ValueError, match="Unknown robot 'so101_leaderr'"):
        Robot("so101_leaderr", mode="sim")
