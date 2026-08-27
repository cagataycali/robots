"""An arm spawned under an id it has no calibration for says so AT SPAWN, not in a log nobody reads.

Measured on the live fleet 2026-08-20: `so101-leader` was spawned as a REAL robot with
robot_id="leader". lerobot looks for robots/so101_follower/leader.json; the only `leader`
calibration on that machine is teleoperators/so101_leader/leader.json, recorded for the teleoperator
side. The bus raised "has no calibration registered", and what the operator could SEE was an arm
publishing presence connected:true with ZERO joints - indistinguishable from a slow probe, a camera
problem or a bus collision, because the reason was buried in a child log at debug level.

The filesystem knows this before the child process does. These tests pin the diagnosis, and pin just
as hard the cases where it must stay SILENT: a wrong guess here would train its reader to ignore it.
"""

from __future__ import annotations

import json

from strands_robots.dashboard.calibration import robot_calibration_gap


def _calib(root, device_type: str, model: str, device_id: str) -> None:
    d = root / device_type / model
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{device_id}.json").write_text(json.dumps({"shoulder_pan": {"id": 1}}))


class TestTheLiveFailure:
    def test_a_teleoperator_calibration_used_as_a_robot_is_named_with_its_consequence(self, tmp_path):
        _calib(tmp_path, "teleoperators", "so101_leader", "leader")
        _calib(tmp_path, "robots", "so101_follower", "follower")
        gap = robot_calibration_gap("so101", "leader", root=tmp_path)
        assert gap and "teleoperator" in gap
        assert "has no calibration registered" in gap, "the words the operator will actually see"
        assert "no joints" in gap, "and the SYMPTOM, so the two can be connected"
        assert "follower" in gap, "plus an id that would work"
        assert str(tmp_path) in gap, "and the file it did find, so the guess is checkable"

    def test_an_id_with_no_calibration_anywhere_lists_the_ones_that_exist(self, tmp_path):
        _calib(tmp_path, "robots", "so101_follower", "follower")
        _calib(tmp_path, "robots", "so101_follower", "follower_arm")
        gap = robot_calibration_gap("so101", "banana", root=tmp_path)
        assert gap and "follower, follower_arm" in gap
        assert "teleoperator" not in gap, "no phantom cross-role explanation when there is none"


class TestSilenceWhereItCannotBeSure:
    """Each of these would be a confident wrong sentence, which is worse than no sentence."""

    def test_the_id_lerobot_will_actually_find_is_not_warned_about(self, tmp_path):
        _calib(tmp_path, "robots", "so101_follower", "follower")
        assert robot_calibration_gap("so101", "follower", root=tmp_path) is None

    def test_a_missing_calibration_cache_says_nothing(self, tmp_path):
        assert robot_calibration_gap("so101", "follower", root=tmp_path / "nope") is None

    def test_a_robot_type_with_no_model_directory_says_nothing(self, tmp_path):
        _calib(tmp_path, "robots", "so101_follower", "follower")
        assert robot_calibration_gap("koch", "follower", root=tmp_path) is None, (
            "an unknown layout must not be described as a missing calibration"
        )

    def test_no_robot_id_means_no_claim(self, tmp_path):
        _calib(tmp_path, "robots", "so101_follower", "follower")
        assert robot_calibration_gap("so101", None, root=tmp_path) is None
        assert robot_calibration_gap("so101", "", root=tmp_path) is None

    def test_a_suffixless_model_directory_counts_too(self, tmp_path):
        """robots/<name>/ is as legal as robots/<name>_follower/ - match, do not hard-code."""
        _calib(tmp_path, "robots", "so101", "follower")
        assert robot_calibration_gap("so101", "follower", root=tmp_path) is None
