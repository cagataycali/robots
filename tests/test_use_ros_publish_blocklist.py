"""Regression tests for use_ros publish topic blocklist."""

from __future__ import annotations

import pytest

from strands_robots.tools.use_ros import _is_publish_blocked


class TestIsPublishBlocked:
    """Pin the blocklist contract: safety-critical topics blocked, others pass."""

    @pytest.mark.parametrize("topic", [
        "/cmd_vel",
        "/cmd_vel_unstamped",
        "/joint_command",
        "/joint_trajectory",
        "/emergency_stop",
        "/e_stop",
        "/motor_enable",
        "/enable_motor",
        "/disable_motor",
        "/navigate_to_pose",
        "/follow_path",
    ])
    def test_safety_critical_topics_blocked(self, topic):
        err = _is_publish_blocked(topic)
        assert err is not None
        assert "blocked" in err

    @pytest.mark.parametrize("topic", [
        "/my_robot/cmd_vel",
        "/ns1/ns2/cmd_vel",
        "/robot_arm/joint_command",
        "/fleet/robot1/emergency_stop",
    ])
    def test_namespaced_topics_blocked(self, topic):
        """Namespace-prefixed forms of blocked topics must also be caught."""
        err = _is_publish_blocked(topic)
        assert err is not None

    @pytest.mark.parametrize("topic", [
        "/my_custom_topic",
        "/robot/status",
        "/diagnostics",
        "/tf",
        "/rosout",
    ])
    def test_non_blocked_topics_pass(self, topic):
        assert _is_publish_blocked(topic) is None

    @pytest.mark.parametrize("topic", [
        "/cmd_vel_evil",
        "/my_robot/cmd_vel_evil",
        "/not_cmd_vel",
        "/foo/notcmd_vel",
        "/joint_trajectory_status",
        "/emergency_stop_status",
    ])
    def test_substring_does_not_match(self, topic):
        """Blocklist must be exact final-segment match, not substring."""
        assert _is_publish_blocked(topic) is None

    def test_multi_segment_blocklist_entry(self):
        """/joint_trajectory_controller/joint_trajectory is in the default list."""
        err = _is_publish_blocked("/joint_trajectory_controller/joint_trajectory")
        assert err is not None

    def test_env_override_disables(self, monkeypatch):
        """Empty STRANDS_ROS2_PUBLISH_BLOCKLIST disables the blocklist entirely."""
        monkeypatch.setenv("STRANDS_ROS2_PUBLISH_BLOCKLIST", "")
        assert _is_publish_blocked("/cmd_vel") is None

    def test_env_override_custom_list(self, monkeypatch):
        monkeypatch.setenv("STRANDS_ROS2_PUBLISH_BLOCKLIST", "/custom_danger,/other_bad")
        assert _is_publish_blocked("/custom_danger") is not None
        # Default blocklist item should now pass
        assert _is_publish_blocked("/cmd_vel") is None
