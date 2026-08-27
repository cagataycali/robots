"""Drift bombs from DASHBOARD_VS_SDK.md §5 — cross-checks that FAIL LOUDLY.

Each dashboard-side copy of an SDK/upstream value is pinned to its source of
truth here, so divergence turns a silent behaviour change into a red test that
names both sides. Measured 2026-08-22 before the fixes:
- mesh_bridge.MAX_CMD_BYTES was a hand copy of 16*1024; the SDK owns
  cmd_bytes_cap() and the transport DROPS anything over it pre-deserialise,
  so a disagreement means "passed the pre-check, vanished into a timeout".
- checkpoints._FAMILY_RE was a hand list missing 6 of lerobot's 19 registered
  policy families, with a `wall_x` entry its own normalizer made unreachable.
- training.SPEC_KEYS mirrors train_policy's signature by hand; a new
  train_policy parameter must show up in SPEC_KEYS or _NOT_IN_FORM by name.

Run with --no-cov (single-file runs trip the global coverage gate).
"""

from __future__ import annotations

import inspect

import pytest


class TestCmdBytesCap:
    def test_mesh_bridge_cap_is_the_sdk_cap(self):
        from strands_robots.dashboard import mesh_bridge
        from strands_robots.mesh._zenoh_config import cmd_bytes_cap

        assert mesh_bridge.MAX_CMD_BYTES == cmd_bytes_cap(), (
            "mesh_bridge's pre-publish size check disagrees with the transport's "
            "drop filter — a command can pass the check and vanish into a timeout"
        )


class TestSpecKeysMirror:
    def test_spec_keys_plus_exclusions_cover_train_policy_exactly(self):
        from strands_robots.dashboard.training import _NOT_IN_FORM, SPEC_KEYS
        from strands_robots.tools.train_policy import train_policy

        params = set(inspect.signature(train_policy).parameters)
        form = set(SPEC_KEYS)
        excluded = set(_NOT_IN_FORM)

        assert form & excluded == set(), "a field cannot be both offered and excluded"
        missing = params - form - excluded
        assert missing == set(), (
            f"train_policy grew parameter(s) {sorted(missing)} that the form neither "
            "offers (SPEC_KEYS) nor excludes with a reason (_NOT_IN_FORM)"
        )
        phantom = (form | excluded) - params
        assert phantom == set(), (
            f"{sorted(phantom)} are not train_policy parameters — a rename upstream "
            "would silently strand the form field"
        )


class TestFamilyMatcher:
    def test_every_registered_lerobot_family_is_guessed(self):
        """The matcher is DERIVED from lerobot's registry, so this can only
        fail if the derivation breaks — which is exactly when it should."""
        lerobot = pytest.importorskip("lerobot.policies")  # noqa: F841
        from lerobot.configs.policies import PreTrainedConfig

        from strands_robots.dashboard.checkpoints import _guess_policy_type

        known = sorted(PreTrainedConfig.get_known_choices())
        assert known, "lerobot registered zero policy families — registry moved?"
        misguessed = {
            k: _guess_policy_type(f"user/{k}_base", [])
            for k in known
            if _guess_policy_type(f"user/{k}_base", []) != k
        }
        assert misguessed == {}, f"registry families the matcher cannot round-trip: {misguessed}"

    def test_underscore_families_match_despite_separator_normalization(self):
        """The old hand regex's `wall_x` literal could never fire: sources are
        normalized `_`->`-` before searching. Pin the tolerant behaviour."""
        from strands_robots.dashboard.checkpoints import _guess_policy_type

        assert _guess_policy_type("org/wall_x_finetune", []) == "wall_x"
        assert _guess_policy_type("org/pi0fast-v2", []) == "pi0_fast"  # no separator
        assert _guess_policy_type("org/pi0-base", []) == "pi0"  # longest-first, not pi0_fast
        assert _guess_policy_type("org/nothing-here", []) is None

    def test_fallback_list_builds_a_working_matcher(self):
        """When lerobot is absent the fallback must still match its own names."""
        from strands_robots.dashboard.checkpoints import (
            _FALLBACK_FAMILIES,
            _build_family_matcher,
        )

        regex, canonical = _build_family_matcher(_FALLBACK_FAMILIES)
        for name in _FALLBACK_FAMILIES:
            m = regex.search(f"user/{name}-base".replace("_", "-"))
            assert m, f"fallback family {name!r} unmatched by its own matcher"
            assert canonical[m.group(1).lower().replace("-", "")] == name
