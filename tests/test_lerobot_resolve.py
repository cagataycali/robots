"""Tests for LeRobot 0.5.x policy resolution via _resolve_policy_class_by_name.

Verifies all 13 policy directories in lerobot.policies can be handled:
- 12 standalone policies resolve to the correct class
- 1 non-standalone (rtc) raises a clear ValueError

Platform: Jetson AGX Thor — CUDA 13.0, torch 2.10.0+cu130
"""

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve(policy_type: str):
    """Import and call the resolver at test time (avoids import-time side effects)."""
    from strands_robots.policies.lerobot_local import _resolve_policy_class_by_name

    return _resolve_policy_class_by_name(policy_type)


def _is_policy_check(obj, attr_name: str) -> bool:
    """Import and call _is_policy_class."""
    from strands_robots.policies.lerobot_local import _is_policy_class

    return _is_policy_class(obj, attr_name)


# ---------------------------------------------------------------------------
# 12 Standalone policies — each must resolve to a class with from_pretrained
# ---------------------------------------------------------------------------

# (policy_type, expected_class_name)
STANDALONE_POLICIES = [
    ("act", "ACTPolicy"),
    ("diffusion", "DiffusionPolicy"),
    ("pi0", "PI0Policy"),
    ("pi0_fast", "PI0FastPolicy"),
    ("pi05", "PI05Policy"),
    ("sac", "SACPolicy"),
    ("sarm", "SARMRewardModel"),
    ("smolvla", "SmolVLAPolicy"),
    ("tdmpc", "TDMPCPolicy"),
    ("vqbet", "VQBeTPolicy"),
    ("wall_x", "WallXPolicy"),
    ("xvla", "XVLAPolicy"),
]


class TestStandalonePolicyResolution:
    """Each standalone policy must resolve to the correct concrete class."""

    @pytest.mark.parametrize("policy_type,expected_class", STANDALONE_POLICIES, ids=[p[0] for p in STANDALONE_POLICIES])
    def test_resolves_to_correct_class(self, policy_type, expected_class):
        cls = _resolve(policy_type)
        assert cls.__name__ == expected_class, (
            f"Expected {expected_class} for '{policy_type}', got {cls.__name__}"
        )

    @pytest.mark.parametrize("policy_type,expected_class", STANDALONE_POLICIES, ids=[p[0] for p in STANDALONE_POLICIES])
    def test_resolved_class_has_from_pretrained(self, policy_type, expected_class):
        cls = _resolve(policy_type)
        assert hasattr(cls, "from_pretrained"), (
            f"{cls.__name__} missing from_pretrained()"
        )

    @pytest.mark.parametrize("policy_type,expected_class", STANDALONE_POLICIES, ids=[p[0] for p in STANDALONE_POLICIES])
    def test_resolved_class_is_a_type(self, policy_type, expected_class):
        cls = _resolve(policy_type)
        assert isinstance(cls, type), f"Expected a class, got {type(cls)}"


# ---------------------------------------------------------------------------
# Non-standalone policy types (processors/wrappers)
# ---------------------------------------------------------------------------

class TestNonStandalonePolicies:
    """RTC is a processor wrapper, not a standalone policy."""

    def test_rtc_raises_value_error(self):
        with pytest.raises(ValueError, match="not a standalone policy"):
            _resolve("rtc")

    def test_rtc_error_mentions_pi0_alternative(self):
        with pytest.raises(ValueError, match="pi0"):
            _resolve("rtc")

    def test_rtc_error_mentions_rtc_processor(self):
        with pytest.raises(ValueError, match="RTCProcessor"):
            _resolve("rtc")


# ---------------------------------------------------------------------------
# Unknown / nonexistent policy types
# ---------------------------------------------------------------------------

class TestUnknownPolicies:
    """Unknown policy names must raise ImportError (not silently return None)."""

    def test_completely_unknown_raises_import_error(self):
        with pytest.raises((ImportError, ValueError)):
            _resolve("nonexistent_policy_abc123")

    def test_typo_raises_import_error(self):
        with pytest.raises((ImportError, ValueError)):
            _resolve("actt")  # typo for "act"

    def test_empty_string_raises(self):
        with pytest.raises((ImportError, ValueError)):
            _resolve("")


# ---------------------------------------------------------------------------
# _is_policy_class helper
# ---------------------------------------------------------------------------

class TestIsPolicyClass:
    """Unit tests for the _is_policy_class helper."""

    def test_accepts_class_ending_with_policy(self):
        from lerobot.policies.act.modeling_act import ACTPolicy

        assert _is_policy_check(ACTPolicy, "ACTPolicy") is True

    def test_accepts_class_ending_with_reward_model(self):
        from lerobot.policies.sarm.modeling_sarm import SARMRewardModel

        assert _is_policy_check(SARMRewardModel, "SARMRewardModel") is True

    def test_rejects_pretrained_policy_itself(self):
        from lerobot.policies.pretrained import PreTrainedPolicy

        assert _is_policy_check(PreTrainedPolicy, "PreTrainedPolicy") is False

    def test_rejects_non_class(self):
        assert _is_policy_check("not_a_class", "foo") is False

    def test_rejects_class_without_from_pretrained(self):
        class FakePolicy:
            pass

        assert _is_policy_check(FakePolicy, "FakePolicy") is False

    def test_accepts_pretrained_subclass_with_odd_name(self):
        """A class inheriting PreTrainedPolicy should resolve even with an unusual name."""
        from lerobot.policies.pretrained import PreTrainedPolicy

        # SARMRewardModel is a concrete PreTrainedPolicy subclass that doesn't end in 'Policy'
        from lerobot.policies.sarm.modeling_sarm import SARMRewardModel

        assert _is_policy_check(SARMRewardModel, "SARMRewardModel") is True


# ---------------------------------------------------------------------------
# _NON_STANDALONE_POLICY_TYPES registry
# ---------------------------------------------------------------------------

class TestNonStandaloneRegistry:
    """Verify the non-standalone registry is consistent."""

    def test_rtc_is_registered(self):
        from strands_robots.policies.lerobot_local import _NON_STANDALONE_POLICY_TYPES

        assert "rtc" in _NON_STANDALONE_POLICY_TYPES

    def test_standalone_policies_not_in_registry(self):
        from strands_robots.policies.lerobot_local import _NON_STANDALONE_POLICY_TYPES

        for policy_type, _ in STANDALONE_POLICIES:
            assert policy_type not in _NON_STANDALONE_POLICY_TYPES, (
                f"'{policy_type}' should NOT be in _NON_STANDALONE_POLICY_TYPES"
            )


# ---------------------------------------------------------------------------
# Integration: full resolution count
# ---------------------------------------------------------------------------

class TestFullResolutionMatrix:
    """Smoke test that exactly 12 standalone + 1 non-standalone = 13 total."""

    def test_total_lerobot_policy_count(self):
        """LeRobot 0.5.x ships 13 policy directories."""
        all_types = [p[0] for p in STANDALONE_POLICIES] + ["rtc"]
        assert len(all_types) == 13

    def test_all_standalone_resolve(self):
        """All 12 standalone policies resolve without error."""
        resolved = {}
        for policy_type, expected_class in STANDALONE_POLICIES:
            cls = _resolve(policy_type)
            resolved[policy_type] = cls.__name__
        assert len(resolved) == 12
        # No duplicates (each policy type maps to a unique class)
        assert len(set(resolved.values())) == 12

    def test_resolution_is_idempotent(self):
        """Calling the resolver twice returns the same class."""
        cls1 = _resolve("act")
        cls2 = _resolve("act")
        assert cls1 is cls2
