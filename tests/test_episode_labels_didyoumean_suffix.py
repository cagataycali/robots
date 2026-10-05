#!/usr/bin/env python3
"""Pin the ``Did you mean 'X'?`` suffix on the four ``episode_labels``
vocabulary refusals (``annotate_episode``, ``filter_episodes``,
``measure_agreement``), and pin that the suffix is additive: a non-match
keeps the byte-exact historical message.

Convention: ``difflib.get_close_matches(cutoff=0.6, n=1)`` - the same ratio
17+ sibling refusals use across the codebase.
"""

from __future__ import annotations

import pytest

from strands_robots.episode_labels import (
    FAILURE_MODES,
    QUALITY_GRADES,
    _didyoumean_suffix,
    annotate_episode,
    filter_episodes,
)


class TestDidYouMeanHelper:
    @pytest.mark.parametrize(
        ("bad", "options", "expected"),
        [
            # QUALITY_GRADES hits
            ("hi", QUALITY_GRADES, " Did you mean 'high'?"),
            ("Medium", QUALITY_GRADES, " Did you mean 'medium'?"),
            # FAILURE_MODES hits
            ("near-miss", FAILURE_MODES, " Did you mean 'near_miss'?"),
            ("occlusion", FAILURE_MODES, " Did you mean 'camera_occlusion'?"),
            ("collide", FAILURE_MODES, " Did you mean 'collision'?"),
            ("camera-occlusion", FAILURE_MODES, " Did you mean 'camera_occlusion'?"),
            # No match at cutoff=0.6 - preserves historical byte-exact message
            ("excellent", QUALITY_GRADES, ""),
            ("good", QUALITY_GRADES, ""),
            ("sloppy", FAILURE_MODES, ""),
            ("jerky", FAILURE_MODES, ""),  # pinned by test_holdout_failure_mode_vocabulary.py
            # Non-str is a safe no-op (empty string, so `+ _didyoumean_suffix(...)` is a no-op)
            (None, QUALITY_GRADES, ""),
            (42, QUALITY_GRADES, ""),
            ([], FAILURE_MODES, ""),
        ],
    )
    def test_helper_mirrors_sibling_refusal_convention(self, bad, options, expected):
        assert _didyoumean_suffix(bad, options) == expected


class TestAnnotateEpisodeHint:
    def test_quality_typo_lands_on_the_vocabulary_grade(self):
        with pytest.raises(ValueError) as exc:
            annotate_episode("/tmp/__nowhere_episode_labels_test", 0, quality="hi")
        text = str(exc.value)
        assert "quality must be one of" in text
        assert "Did you mean 'high'?" in text

    def test_quality_case_mismatch_lands_on_canonical_form(self):
        with pytest.raises(ValueError) as exc:
            annotate_episode("/tmp/__nowhere_episode_labels_test", 0, quality="Medium")
        assert "Did you mean 'medium'?" in str(exc.value)

    @pytest.mark.parametrize(
        ("typo", "canonical"),
        [
            ("near-miss", "near_miss"),
            ("camera-occlusion", "camera_occlusion"),
            ("occlusion", "camera_occlusion"),
            ("collide", "collision"),
        ],
    )
    def test_failure_mode_dash_vs_underscore_or_short_form_lands(self, typo, canonical):
        with pytest.raises(ValueError) as exc:
            annotate_episode(
                "/tmp/__nowhere_episode_labels_test", 0, quality="high", failure_mode=typo
            )
        text = str(exc.value)
        assert "failure_mode must be None or one of" in text
        assert f"Did you mean {canonical!r}?" in text

    def test_a_far_typo_keeps_the_byte_exact_historical_message(self):
        """Pin the additive invariant: no close match -> no suffix, old message intact."""
        with pytest.raises(ValueError) as exc:
            annotate_episode("/tmp/__nowhere_episode_labels_test", 0, quality="excellent")
        assert str(exc.value) == (
            f"annotate_episode: quality must be one of {QUALITY_GRADES}, got 'excellent'."
        )


class TestFilterEpisodesHint:
    def test_min_quality_near_typo_gets_the_hint(self):
        with pytest.raises(ValueError) as exc:
            filter_episodes("/tmp/__nowhere_episode_labels_test", min_quality="Medium")
        assert "Did you mean 'medium'?" in str(exc.value)

    def test_min_quality_far_typo_keeps_byte_exact_historical_message(self):
        with pytest.raises(ValueError) as exc:
            filter_episodes("/tmp/__nowhere_episode_labels_test", min_quality="excellent")
        assert str(exc.value) == (
            f"filter_episodes: min_quality must be one of {QUALITY_GRADES}, got 'excellent'."
        )
