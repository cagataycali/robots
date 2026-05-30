"""Tests for ``strands_robots.policies.lerobot_local.resolution`` -- the
LeRobot policy class lookup that ``LerobotLocalPolicy`` uses to turn a
HuggingFace Hub repo id into a concrete ``PreTrainedPolicy`` subclass."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skipif(
    pytest.importorskip("lerobot", reason="lerobot is required for resolution tests") is None,
    reason="lerobot not installed",
)


class TestPolicyConfigDiscovery:
    """Regression tests for ``_ensure_policy_configs_registered()``.

    The previous implementation imported a single hand-coded canary
    (``lerobot.policies.act.configuration_act``) and assumed lerobot's
    eager ``policies/__init__.py`` would side-effect every other policy
    config into the draccus ``PreTrainedConfig`` registry. That breaks
    the moment lerobot makes its policies subpackage lazy (the same
    transition ``lerobot.robots`` already went through), and it also
    breaks today inside ``LerobotLocalPolicy`` because that path
    intentionally installs a stub for ``lerobot.policies`` (to skip
    eagerly importing transformers/flash-attn dependencies of unrelated
    policies like groot).
    """

    def test_pkgutil_walk_registers_every_lerobot_policy_subpackage(self):
        """Walking ``lerobot.policies`` with pkgutil registers every
        policy config without any hand-coded list. The discovery is
        symmetric with the robots-side fix in
        ``hardware_robot._ensure_lerobot_robots_registered``.
        """
        from lerobot.configs.policies import PreTrainedConfig

        from strands_robots.policies.lerobot_local.resolution import (
            _ensure_policy_configs_registered,
        )

        _ensure_policy_configs_registered.cache_clear()
        _ensure_policy_configs_registered()

        registered = set(PreTrainedConfig.get_known_choices().keys())

        # All policies that ship in lerobot >=0.5.x. Adding more upstream
        # is a no-op for strands_robots -- the pkgutil walker picks them
        # up automatically.
        expected_min = {
            "act",
            "diffusion",
            "pi0",
            "smolvla",
            "tdmpc",
            "vqbet",
            "molmoact2",  # lerobot PR #3604, shipped in 0.5.2+
        }
        missing = expected_min - registered
        assert not missing, f"Discovery missed lerobot built-in policies: {missing}. Registered: {sorted(registered)}"

    def test_molmoact2_registered_after_stubbed_lerobot_policies(self):
        """The ``LerobotLocalPolicy`` runtime path installs a lightweight
        stub for ``lerobot.policies`` (to avoid executing its potentially
        heavy ``__init__.py`` that pulls in transformers/flash-attn).
        Even with that stub in place -- which short-circuits any
        side-effect-on-init style registration -- ``molmoact2`` and
        every other lerobot built-in policy must still resolve.

        Pre-fix, the stub combined with the single-canary import meant
        ONLY ``act`` ended up registered; lookups for any other policy
        type silently fell through to manual config.json parsing,
        which failed for repos that rely on draccus resolution.
        """
        import sys

        # Reset every cached lerobot import so the stub gets a chance
        # to take effect on this test.
        for mod_name in [m for m in sys.modules if "lerobot" in m]:
            del sys.modules[mod_name]

        from strands_robots.policies.lerobot_local.resolution import (
            _ensure_lerobot_policies_importable,
            _ensure_policy_configs_registered,
        )

        _ensure_lerobot_policies_importable()  # installs the stub
        _ensure_policy_configs_registered.cache_clear()
        _ensure_policy_configs_registered()

        from lerobot.configs.policies import PreTrainedConfig

        registered = set(PreTrainedConfig.get_known_choices().keys())
        assert "molmoact2" in registered, (
            f"molmoact2 missing after stub+walk; registered: {sorted(registered)}. "
            "Did the pkgutil walker get reverted to single-canary bootstrap?"
        )
        # Also verify the symmetric case for an older policy that pre-dates
        # the stub mechanism, to make sure we didn't break the existing path.
        assert "act" in registered

    def test_resolve_class_by_name_handles_molmoact2_modeling_convention(self):
        """``modeling_<type>`` lookup works for new policies that follow
        the convention. molmoact2's class lives at
        ``lerobot.policies.molmoact2.modeling_molmoact2.MolmoAct2Policy``;
        this path is the second strategy after the draccus registry."""
        pytest.importorskip("lerobot.policies.molmoact2.modeling_molmoact2")
        from strands_robots.policies.lerobot_local.resolution import (
            resolve_policy_class_by_name,
        )

        cls = resolve_policy_class_by_name("molmoact2")
        assert cls.__name__ == "MolmoAct2Policy"
        assert cls.__module__.endswith("molmoact2.modeling_molmoact2")
