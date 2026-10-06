"""Pin: `register_robot` warns on an unknown hardware field, mirroring category.

:func:`_warn_on_a_near_miss_category` emits a WARNING on a near-miss category
(``"arms"`` for ``"arm"``) so the user is told the one-robot silo their typo
opened. :func:`_warn_on_unknown_hardware_fields` is its sibling for the
``hardware`` block schema (``driver``, ``lerobot_type``,
``requires_lerobot_from_source``): a typo'd key is dead weight at best
and a silent misconfiguration at worst (``"lerobot_tpye"`` for
``"lerobot_type"`` passes the asset-less declaration check if a sibling
``driver="strands"`` carries it, then the typed value is never read).

Pinned here, next to :file:`test_register_robot_without_an_asset.py`, so a
future refactor cannot drop one half of the helper pair.
"""
from __future__ import annotations

import logging
import pytest


@pytest.fixture
def isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("STRANDS_BASE_DIR", str(tmp_path))
    assets = tmp_path / "assets"
    assets.mkdir(exist_ok=True)
    monkeypatch.setenv("STRANDS_ASSETS_DIR", str(assets))
    yield


def _register_capture_warnings(caplog, **kwargs):
    """Call register_robot inside caplog and return (entry, hw_warnings)."""
    from strands_robots.registry import register_robot, unregister_robot

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="strands_robots.registry.user_registry"):
        entry = register_robot(**kwargs)
    hw_msgs = [r.getMessage() for r in caplog.records if "hardware" in r.getMessage()]
    unregister_robot(kwargs["name"])
    return entry, hw_msgs


class TestHardwareFieldWarns:
    """Mirror of :file:`test_register_robot_category_did_you_mean` (removed)."""

    @pytest.mark.parametrize(
        "typo,expected_hint",
        [
            ("lerobot_tpye", "lerobot_type"),  # one-char swap
            ("lerobot_typ", "lerobot_type"),  # trailing drop
            ("drvier", "driver"),  # transposition
            ("requires_lerobot", "requires_lerobot_from_source"),  # truncation
        ],
    )
    def test_near_miss_hardware_key_is_warned_with_did_you_mean(
        self, isolate, caplog, typo, expected_hint
    ):
        _, hw_msgs = _register_capture_warnings(
            caplog,
            name=f"t_{typo}",
            hardware={"driver": "strands", typo: "value"},
        )
        assert len(hw_msgs) == 1, (
            f"expected one hardware WARNING for typo {typo!r}, got {hw_msgs!r}"
        )
        assert f"hardware.{typo}" in hw_msgs[0]
        assert expected_hint in hw_msgs[0], (
            f"expected hint for {typo!r} to name {expected_hint!r}, got {hw_msgs[0]!r}"
        )

    def test_unknown_key_without_a_near_miss_is_still_warned(self, isolate, caplog):
        _, hw_msgs = _register_capture_warnings(
            caplog,
            name="t_junk",
            hardware={"driver": "strands", "xyzzy": "nope"},
        )
        assert len(hw_msgs) == 1
        assert "hardware.xyzzy" in hw_msgs[0]
        # No close match -> enumerate known fields
        assert "driver" in hw_msgs[0] and "lerobot_type" in hw_msgs[0]

    def test_canonical_keys_do_not_warn(self, isolate, caplog):
        _, hw_msgs = _register_capture_warnings(
            caplog,
            name="t_clean",
            hardware={"driver": "strands"},
        )
        assert hw_msgs == []

    def test_full_canonical_block_does_not_warn(self, isolate, caplog):
        _, hw_msgs = _register_capture_warnings(
            caplog,
            name="t_full",
            hardware={
                "driver": "lerobot",
                "lerobot_type": "koch",
                "requires_lerobot_from_source": False,
            },
        )
        assert hw_msgs == []

    def test_declaration_failure_still_warns_before_raising(self, isolate, caplog):
        """The warn runs at the top of register_robot, before the required-hardware check.

        A user whose declaration ALSO fails (no `lerobot_type`, no
        `driver='strands'`) should see the hint fired as a WARNING before
        the ValueError dumps the dict - two signals, not one.
        """
        from strands_robots.registry import register_robot

        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="strands_robots.registry.user_registry"):
            with pytest.raises(ValueError, match="hardware must"):
                register_robot(
                    name="t_dualfail",
                    hardware={"driver": "lerobot", "lerobot_tpye": "koch"},
                )
        hw_msgs = [r.getMessage() for r in caplog.records if "hardware.lerobot_tpye" in r.getMessage()]
        assert len(hw_msgs) == 1
        assert "lerobot_type" in hw_msgs[0]

    def test_non_dict_hardware_does_not_warn(self, isolate, caplog):
        """`hardware=None` is the default (asset-based registration); no warn."""
        # Have to provide an asset to skip _require_hardware_declaration.
        import os

        base = os.environ["STRANDS_ASSETS_DIR"]
        asset_dir = os.path.join(base, "t_none_hw")
        os.makedirs(asset_dir, exist_ok=True)
        with open(os.path.join(asset_dir, "m.xml"), "w") as f:
            f.write("<mujoco/>")

        from strands_robots.registry import register_robot, unregister_robot

        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="strands_robots.registry.user_registry"):
            register_robot(name="t_none_hw", model_xml="m.xml", asset_dir="t_none_hw")
        hw_msgs = [r.getMessage() for r in caplog.records if "hardware" in r.getMessage()]
        assert hw_msgs == []
        unregister_robot("t_none_hw")
