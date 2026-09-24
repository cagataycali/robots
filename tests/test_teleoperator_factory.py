"""Tests for the Teleoperator() factory - config resolution without hardware.

These build lerobot ``TeleoperatorConfig`` objects and (for config-only
assertions) never connect to a device, so they run on CI without USB.
"""

from __future__ import annotations

import dataclasses

import pytest

from strands_robots.teleoperator import Teleoperator, _build_teleop_config
from strands_robots.utils import ensure_lerobot_family_registered

pytest.importorskip("lerobot", reason="factory tests require lerobot installed")


def test_registry_walk_populates_known_choices():
    from lerobot.teleoperators.config import TeleoperatorConfig

    ensure_lerobot_family_registered("teleoperators")
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


# ---------------------------------------------------------------------------
# kwarg-filtering + registry-failure edge paths (no hardware, no device SDKs)
#
# These exercise the cross-device polymorphism carve-out and the
# registry/import failure modes by substituting a synthetic draccus config
# class, so they hold regardless of which device SDKs are installed.
# ---------------------------------------------------------------------------


def test_build_config_forwards_declared_field_and_drops_undeclared_allowlist():
    """A dataclass-declared-but-not-allowlisted kwarg is forwarded; an
    allowlisted kwarg the dataclass does not declare is silently dropped
    (cross-device polymorphism), not rejected as a typo."""
    from lerobot.teleoperators.config import TeleoperatorConfig

    @dataclasses.dataclass
    class _FakeConfig:
        id: str | None = None
        custom_field: int = 0  # declared but NOT on the allowlist

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            TeleoperatorConfig,
            "get_choice_class",
            classmethod(lambda cls, t: _FakeConfig),
        )
        cfg = _build_teleop_config("fake", custom_field=42, baud_rate=9600)

    assert cfg.custom_field == 42  # declared field forwarded
    assert not hasattr(cfg, "baud_rate")  # allowlisted-but-undeclared dropped


def test_build_config_non_dataclass_raises_typeerror():
    """A non-dataclass config class cannot be kwarg-filtered safely -> TypeError."""
    from lerobot.teleoperators.config import TeleoperatorConfig

    class _NotADataclass:
        def __init__(self, **kwargs):  # noqa: D401 - test stub
            pass

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            TeleoperatorConfig,
            "get_choice_class",
            classmethod(lambda cls, t: _NotADataclass),
        )
        with pytest.raises(TypeError, match="non-dataclass config class"):
            _build_teleop_config("weird", port="/dev/ttyACM0")


def test_build_config_construction_failure_raises_valueerror():
    """A dataclass that rejects its own values surfaces as a clean ValueError
    naming the config class, not a raw __post_init__ traceback."""
    from lerobot.teleoperators.config import TeleoperatorConfig

    @dataclasses.dataclass
    class _BadConfig:
        id: str | None = None
        port: str | None = None

        def __post_init__(self):
            raise ValueError("boom")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            TeleoperatorConfig,
            "get_choice_class",
            classmethod(lambda cls, t: _BadConfig),
        )
        with pytest.raises(ValueError, match=r"Failed to construct _BadConfig.*boom"):
            _build_teleop_config("bad", port="/dev/ttyACM0")
