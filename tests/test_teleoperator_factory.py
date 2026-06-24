"""Tests for the Teleoperator() factory - config resolution without hardware.

These build lerobot ``TeleoperatorConfig`` objects and (for config-only
assertions) never connect to a device, so they run on CI without USB.
"""

from __future__ import annotations

import pytest

from strands_robots.teleoperator import (
    Teleoperator,
    _build_teleop_config,
    _ensure_lerobot_teleoperators_registered,
)

pytest.importorskip("lerobot", reason="factory tests require lerobot installed")


def test_registry_walk_populates_known_choices():
    from lerobot.teleoperators.config import TeleoperatorConfig

    _ensure_lerobot_teleoperators_registered()
    choices = set(TeleoperatorConfig.get_known_choices())
    # A representative slice that must always be present.
    assert {"so101_leader", "so100_leader", "gamepad", "keyboard"} <= choices


def test_build_config_forwards_port_and_id():
    cfg = _build_teleop_config("so101_leader", port="/dev/ttyACM1", id="blue")
    assert cfg.port == "/dev/ttyACM1"
    assert cfg.id == "blue"


def test_build_config_gamepad_field():
    cfg = _build_teleop_config("gamepad", use_gripper=False)
    assert cfg.use_gripper is False


def test_build_config_rejects_typo_kwarg():
    with pytest.raises(ValueError, match=r"Unknown kwarg.*prot"):
        _build_teleop_config("so101_leader", prot="/dev/ttyACM1")


def test_build_config_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unsupported teleoperator type"):
        _build_teleop_config("totally_made_up_leader")


def test_factory_rejects_empty_name():
    with pytest.raises(ValueError, match="Invalid teleoperator name"):
        Teleoperator("")


def test_factory_builds_instance():
    """Full factory path -> a real lerobot Teleoperator instance (not connected).

    Instantiating a concrete leader can require an optional device SDK (e.g.
    ``feetech-servo-sdk`` for SO arms). When that SDK is absent the factory
    still did its job - it resolved the config and delegated to lerobot's
    ``make_teleoperator_from_config``, which is where the ImportError surfaces.
    We treat that as a pass (the factory contract held) and only assert the
    duck-typed interface when construction actually succeeds.
    """
    try:
        dev = Teleoperator("so101_leader", port="/dev/ttyACM9", id="testarm")
    except ImportError as exc:
        # Optional servo SDK missing -> factory delegated correctly. Acceptable.
        assert "required but not installed" in str(exc) or "sdk" in str(exc).lower()
        return
    # Duck-typed contract used by TeleopMixin / InputPublisher.
    assert callable(getattr(dev, "get_action", None))
    assert callable(getattr(dev, "connect", None))
    assert hasattr(dev, "is_connected")
    assert dev.id == "testarm"
