"""The \"uncalibrated\" remedy, narrowed to the arm that is actually failing.

The broad remedy (`calibration_advice`) already refuses to send a calibrated arm back to the
teaching pendant — but on this machine it prints TEN paths across three robot families and leaves the
operator to work out which one lerobot wanted. The child knows its own `robot_name` and `robot_id`, so
the answer can be one path and a short list of ids instead.

Measured against the live fleet the day these tests were written: so101-leader, spawned real with
robot_id="leader", produced exactly the \"calibrated for the OTHER side of the pair\" sentence.
"""

from __future__ import annotations

from strands_robots.dashboard.joint_silence import calibration_advice

# The real layout on cagatay's Mac, as _calibrations_on_disk() returns it.
DISK = {
    "robots/earthrover_mini_plus": [],
    "robots/so100_follower": ["follower_arm"],
    "robots/so101_follower": ["follower", "follower_arm", "leader_arm"],
    "robots/so_follower": ["follower", "follower_arm", "leader_arm"],
    "teleoperators/so101_leader": ["leader", "leader_arm"],
}


class TestTheArmThatActuallyFailed:
    def test_an_id_calibrated_for_the_other_side_is_named_as_a_filename_problem(self):
        a = calibration_advice(DISK, robot_name="so101", robot_id="leader")
        assert "robots/so101_follower/leader.json" in a, "the path lerobot wanted, in full"
        assert "teleoperators/so101_leader/leader.json" in a, "and the one that exists"
        assert "Do NOT recalibrate" in a, "the physical work this advice exists to prevent"
        assert "follower, follower_arm, leader_arm" in a, "ids that would work"
        assert "so_follower" not in a, "another robot family is noise here"
        assert "so100" not in a

    def test_a_file_that_exists_stops_the_calibration_story_entirely(self):
        a = calibration_advice(DISK, robot_name="so101", robot_id="follower")
        assert "EXISTS" in a and "not a missing calibration" in a
        assert "recalibrat" not in a.lower(), "never suggest re-teaching an arm that IS calibrated"
        assert "devices > logs" in a, "point at the real exception instead"

    def test_an_id_nothing_has_calibrated_lists_the_ones_for_this_robot_only(self):
        a = calibration_advice(DISK, robot_name="so101", robot_id="banana")
        assert "robots/so101_follower/banana.json" in a
        assert "follower, follower_arm, leader_arm" in a
        assert "OTHER side" not in a, "no phantom cross-role explanation when there is none"


class TestFallingBackInsteadOfGuessing:
    """Each of these must reach the BROAD remedy, which is honest, rather than a narrow guess."""

    def _broad(self, a: str | None) -> bool:
        return bool(a) and "Calibration files DO exist on this machine" in a  # type: ignore[operator]

    def test_an_unknown_robot_family_falls_back(self):
        assert self._broad(calibration_advice(DISK, robot_name="koch", robot_id="follower"))

    def test_no_identity_at_all_falls_back(self):
        assert self._broad(calibration_advice(DISK))
        assert self._broad(calibration_advice(DISK, robot_name="so101"))
        assert self._broad(calibration_advice(DISK, robot_id="leader"))

    def test_nothing_on_disk_still_says_nothing(self):
        assert calibration_advice({}, robot_name="so101", robot_id="leader") is None
        assert calibration_advice(None, robot_name="so101", robot_id="leader") is None
