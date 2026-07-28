# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""A sim home pose outside the checkpoint's state range must be diagnosable.

The SO sim embodiments declare ``state_units="degrees"``, and
``_convert_joint_vector`` implements lerobot's MID-POINT-CENTERED DEGREES mode.
No embodiment in ``embodiments.json`` sets ``joint_mids`` (grep count: 0), so every
conversion uses mid=0 - i.e. it assumes the MJCF's kinematic zero coincides with
the physical arm's calibration mid. It does not.

Because ``q01_q99`` / ``min_max`` normalizers CLIP, an out-of-range dimension does
not merely scale oddly: it collapses to exactly +/-1 and its proprioceptive
information is destroyed, silently. Measured against MolmoAct2's SHIPPED state
stats, whose q01 for joints 2 and 3 is +43.7 and +38.4 degrees - the training data
never contains a near-zero reading for those joints::

    home qpos=0 -> packed     [0, 0, 0, 0, 0, 9.1]
                -> normalized [-0.071, -1.0, -1.0, -1.0, 0.193, -0.622]
                   SATURATED: 3 of 6 dims

This is the diagnosable-instead-of-silent half of the fix. Correcting the offset
needs a per-embodiment sim-vs-hardware zero delta, which is a data change per robot
AND per checkpoint, so it is deliberately not guessed.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import patch

from strands_robots.policies.lerobot_local.norm_stats import FeatureNormalizer
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

# MolmoAct2's real shipped q01/q99 for the SO-100/101 tag.
_Q01 = [-41.9, 43.7, 38.4, 5.7, -63.4, 0.9]
_Q99 = [48.3, 185.3, 173.1, 91.8, 42.9, 44.1]


class TestSaturatingDims:
    def test_the_home_pose_saturates_three_dims(self):
        """Regression evidence, as a unit assertion."""
        normalizer = FeatureNormalizer.from_stats({"q01": _Q01, "q99": _Q99}, "q01_q99")

        assert normalizer.saturating_dims([0.0, 0.0, 0.0, 0.0, 0.0, 9.1]) == [1, 2, 3]

    def test_an_in_range_vector_saturates_nothing(self):
        normalizer = FeatureNormalizer.from_stats({"q01": _Q01, "q99": _Q99}, "q01_q99")

        assert normalizer.saturating_dims([0.0, 100.0, 100.0, 50.0, 0.0, 20.0]) == []

    def test_min_max_mode_also_clips(self):
        normalizer = FeatureNormalizer.from_stats({"min": [0.0] * 3, "max": [1.0] * 3}, "min_max")

        assert normalizer.saturating_dims([-1.0, 0.5, 2.0]) == [0, 2]

    def test_mean_std_has_no_bounds_to_exceed(self):
        """Only the clipping modes can destroy information this way."""
        normalizer = FeatureNormalizer.from_stats({"mean": [0.0] * 3, "std": [1.0] * 3}, "mean_std")

        assert normalizer.saturating_dims([99.0, 99.0, 99.0]) == []

    def test_a_shorter_vector_is_handled(self):
        normalizer = FeatureNormalizer.from_stats({"q01": _Q01, "q99": _Q99}, "q01_q99")

        assert normalizer.saturating_dims([0.0, 0.0]) == [1]

    def test_a_non_numeric_input_is_not_a_crash(self):
        normalizer = FeatureNormalizer.from_stats({"q01": _Q01, "q99": _Q99}, "q01_q99")

        assert normalizer.saturating_dims("not a vector") == []


def _stats_payload(q01: list[float], q99: list[float]) -> dict:
    return {
        "format": "molmoact2_norm_stats.v1",
        "norm_mode": "q01_q99",
        "metadata_by_tag": {"test_tag": {"state_stats": {"q01": q01, "q99": q99}}},
    }


def _policy_with_stats(tmp_path, payload: dict) -> LerobotLocalPolicy:
    (tmp_path / "norm_stats.json").write_text(json.dumps(payload))
    with patch.object(LerobotLocalPolicy, "_load_model"):
        policy = LerobotLocalPolicy(pretrained_name_or_path=str(tmp_path), norm_tag="test_tag")
    return policy


class TestTheLoadTimeWarning:
    def test_it_fires_and_names_the_saturated_joints(self, tmp_path, caplog):
        from strands_robots.policies.lerobot_local.embodiment import load_embodiment

        policy = _policy_with_stats(tmp_path, _stats_payload(_Q01, _Q99))

        with caplog.at_level(logging.WARNING):
            policy._warn_if_home_state_is_out_of_distribution(load_embodiment("so101"))

        messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("OUTSIDE this checkpoint's state distribution" in m for m in messages), messages
        joined = " ".join(messages)
        assert "3 of 6" in joined
        assert "saturate" in joined

    def test_it_is_silent_when_the_home_pose_is_in_range(self, tmp_path, caplog):
        from strands_robots.policies.lerobot_local.embodiment import load_embodiment

        # A range that comfortably contains the packed home vector.
        payload = _stats_payload([-100.0] * 6, [100.0] * 6)
        policy = _policy_with_stats(tmp_path, payload)

        with caplog.at_level(logging.WARNING):
            policy._warn_if_home_state_is_out_of_distribution(load_embodiment("so101"))

        assert not [r for r in caplog.records if "state distribution" in r.getMessage()]

    def test_a_native_units_embodiment_is_never_checked(self, tmp_path, caplog):
        """so_real packs RAW hardware degrees; there is no sim zero to mismatch."""
        from strands_robots.policies.lerobot_local.embodiment import load_embodiment

        policy = _policy_with_stats(tmp_path, _stats_payload(_Q01, _Q99))

        with caplog.at_level(logging.WARNING):
            policy._warn_if_home_state_is_out_of_distribution(load_embodiment("so_real"))

        assert not [r for r in caplog.records if "state distribution" in r.getMessage()]

    def test_the_message_is_plain_ascii(self, tmp_path, caplog):
        """AGENTS.md: user-facing strings are plain ASCII only."""
        from strands_robots.policies.lerobot_local.embodiment import load_embodiment

        policy = _policy_with_stats(tmp_path, _stats_payload(_Q01, _Q99))

        with caplog.at_level(logging.WARNING):
            policy._warn_if_home_state_is_out_of_distribution(load_embodiment("so101"))

        for record in caplog.records:
            assert record.getMessage().isascii()

    def test_a_missing_norm_stats_file_is_not_an_error(self, tmp_path):
        from strands_robots.policies.lerobot_local.embodiment import load_embodiment

        with patch.object(LerobotLocalPolicy, "_load_model"):
            policy = LerobotLocalPolicy(pretrained_name_or_path=str(tmp_path))

        # No norm_stats.json written: the check must simply not fire.
        policy._warn_if_home_state_is_out_of_distribution(load_embodiment("so101"))

    def test_a_broken_stats_payload_never_breaks_the_load(self, tmp_path):
        from strands_robots.policies.lerobot_local.embodiment import load_embodiment

        policy = _policy_with_stats(tmp_path, {"norm_mode": "q01_q99", "metadata_by_tag": {"test_tag": {}}})

        policy._warn_if_home_state_is_out_of_distribution(load_embodiment("so101"))


class TestTheOffsetIsDeliberatelyNotGuessed:
    def test_no_embodiment_declares_joint_mids(self):
        """Pins the current state so a future offset change is a conscious one.

        The ledger's verifier established ``joint_mids`` is NOT the right knob (it
        is the calibration mid, not the sim-vs-hardware zero delta), so this
        asserts the field stays empty rather than being filled with the wrong
        quantity.
        """
        from pathlib import Path

        import strands_robots.policies.lerobot_local as pkg

        data = json.loads((Path(pkg.__file__).parent / "embodiments.json").read_text())
        for name, config in data.get("configs", {}).items():
            assert not config.get("joint_mids"), f"{name} declares joint_mids: {config['joint_mids']}"
