"""A ``category`` close to a known group gets a "Did you mean" warning.

``register_robot(category=...)`` accepted any string without surfacing a near
miss, so a typo like ``"arms"`` would silently open a one-robot silo beside
the registry's real ``"arm"`` group - visible in
:func:`~strands_robots.registry.list_robots_by_category` and in the CLI
table, invisible to the docs filter row's ``data-family="arm"`` button.

:func:`strands_robots.registry.user_registry._warn_unknown_category` closes
the gap: an unknown spelling close to one of the eight categories the
registry and the docs filter row already share emits a WARNING log naming
the typo and the ``difflib`` suggestion. Not a refusal - a caller may
deliberately open a new group (``"quadruped"``) and
:func:`list_robots_by_category` keeps grouping it - just the "Did you mean"
nudge the sibling ``_unknown_robot_msg`` and ``create_policy`` 404 already
make for robot and provider names.
"""

from __future__ import annotations

import logging

import pytest

from strands_robots.registry import register_robot
from strands_robots.registry.user_registry import _KNOWN_CATEGORIES


@pytest.fixture
def user_overlay(tmp_path, monkeypatch):
    """Fresh user-registry overlay rooted in a tmpdir, no asset-dir required."""
    monkeypatch.setenv("STRANDS_BASE_DIR", str(tmp_path))
    yield tmp_path


class TestRegisterRobotCategoryHint:
    def test_known_category_is_silent(self, user_overlay, caplog):
        with caplog.at_level(logging.WARNING, logger="strands_robots.registry.user_registry"):
            register_robot(
                name="my_arm",
                hardware={"driver": "strands"},
                category="arm",  # already in _KNOWN_CATEGORIES
                joints=6,
            )
        # Any WARNINGs from this logger must not be the category-hint message.
        category_warnings = [
            r for r in caplog.records
            if r.name == "strands_robots.registry.user_registry"
            and r.levelno == logging.WARNING
            and "category" in r.getMessage()
        ]
        assert category_warnings == [], (
            f"a known category should not warn, got {[r.getMessage() for r in category_warnings]}"
        )

    def test_typo_category_emits_did_you_mean(self, user_overlay, caplog):
        with caplog.at_level(logging.WARNING, logger="strands_robots.registry.user_registry"):
            register_robot(
                name="my_arm_typo",
                hardware={"driver": "strands"},
                category="arms",  # typo of "arm"
                joints=6,
            )
        hints = [
            r for r in caplog.records
            if r.name == "strands_robots.registry.user_registry"
            and r.levelno == logging.WARNING
            and "'arms'" in r.getMessage()
            and "Did you mean" in r.getMessage()
            and "arm" in r.getMessage()
        ]
        assert hints, (
            "typo 'arms' should warn with a 'Did you mean: arm' hint; "
            f"saw {[(r.name, r.getMessage()) for r in caplog.records]}"
        )

    def test_compound_typo_names_both_close_matches(self, user_overlay, caplog):
        with caplog.at_level(logging.WARNING, logger="strands_robots.registry.user_registry"):
            register_robot(
                name="my_mm",
                hardware={"driver": "strands"},
                category="mobile-manip",  # dash instead of underscore
                joints=6,
            )
        hinted = any(
            "mobile_manip" in r.getMessage() and "Did you mean" in r.getMessage()
            for r in caplog.records
            if r.name == "strands_robots.registry.user_registry" and r.levelno == logging.WARNING
        )
        assert hinted, "'mobile-manip' should hint 'mobile_manip'"

    def test_far_category_warns_but_no_hint(self, user_overlay, caplog):
        """A deliberately new category group warns, but suggests nothing."""
        with caplog.at_level(logging.WARNING, logger="strands_robots.registry.user_registry"):
            register_robot(
                name="my_rover",
                hardware={"driver": "strands"},
                category="quadruped",  # not close to any known category
                joints=12,
            )
        # Exactly one WARNING, and it does NOT say "Did you mean".
        warnings = [
            r for r in caplog.records
            if r.name == "strands_robots.registry.user_registry" and r.levelno == logging.WARNING
        ]
        assert len(warnings) == 1
        assert "Did you mean" not in warnings[0].getMessage()
        # Still names the typo and the group vocabulary so the caller can decide.
        assert "'quadruped'" in warnings[0].getMessage()

    def test_empty_category_is_silent(self, user_overlay, caplog):
        """Empty / whitespace category is reserved for `_UNCATEGORIZED` grouping."""
        with caplog.at_level(logging.WARNING, logger="strands_robots.registry.user_registry"):
            register_robot(
                name="my_blank",
                hardware={"driver": "strands"},
                category="",  # reserved sentinel for "uncategorized"
                joints=0,
            )
            register_robot(
                name="my_ws",
                hardware={"driver": "strands"},
                category="   ",  # whitespace-only, folds to empty
                joints=0,
            )
        hints = [
            r for r in caplog.records
            if r.name == "strands_robots.registry.user_registry"
            and r.levelno == logging.WARNING
            and "category" in r.getMessage()
        ]
        assert hints == [], f"empty/whitespace category must not warn, got {[r.getMessage() for r in hints]}"

    def test_known_categories_match_docs_filter_row(self):
        """The pool is the docs filter row + ``_CATEGORY_DISPLAY_ORDER`` keys."""
        # Spelled explicitly here (not imported) so a change to either side
        # fails this test, which is the point of pinning them together.
        expected = {"arm", "bimanual", "hand", "humanoid", "expressive", "mobile", "mobile_manip", "aerial"}
        assert set(_KNOWN_CATEGORIES) == expected
