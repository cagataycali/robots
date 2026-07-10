"""Regression tests for use_ros publish topic blocklist + HIL gate."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from strands_robots.tools.use_ros import (
    _approve_response,
    _gate_publish,
    _is_publish_blocked,
)


class TestIsPublishBlocked:
    """Pin the blocklist contract: safety-critical topics blocked, others pass."""

    @pytest.mark.parametrize(
        "topic",
        [
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
        ],
    )
    def test_safety_critical_topics_blocked(self, topic):
        err = _is_publish_blocked(topic)
        assert err is not None
        assert "blocked" in err

    @pytest.mark.parametrize(
        "topic",
        [
            "/my_robot/cmd_vel",
            "/ns1/ns2/cmd_vel",
            "/robot_arm/joint_command",
            "/fleet/robot1/emergency_stop",
        ],
    )
    def test_namespaced_topics_blocked(self, topic):
        """Namespace-prefixed forms of blocked topics must also be caught."""
        err = _is_publish_blocked(topic)
        assert err is not None

    @pytest.mark.parametrize(
        "topic",
        [
            "/my_custom_topic",
            "/robot/status",
            "/diagnostics",
            "/tf",
            "/rosout",
        ],
    )
    def test_non_blocked_topics_pass(self, topic):
        assert _is_publish_blocked(topic) is None

    @pytest.mark.parametrize(
        "topic",
        [
            "/cmd_vel_evil",
            "/my_robot/cmd_vel_evil",
            "/not_cmd_vel",
            "/foo/notcmd_vel",
            "/joint_trajectory_status",
            "/emergency_stop_status",
        ],
    )
    def test_substring_does_not_match(self, topic):
        """Blocklist must be exact final-segment match, not substring."""
        assert _is_publish_blocked(topic) is None

    def test_multi_segment_blocklist_entry(self):
        """/joint_trajectory_controller/joint_trajectory is in the default list."""
        err = _is_publish_blocked("/joint_trajectory_controller/joint_trajectory")
        assert err is not None


class TestGatePublish:
    """Pin the HIL gate contract: allowlist, bypass, interrupt, decline."""

    @pytest.fixture(autouse=True)
    def _hermetic_gate_env(self, monkeypatch):
        """Neutralize ambient env that short-circuits the gate.

        Both BYPASS_TOOL_CONSENT and STRANDS_ROS2_PUBLISH_ALLOW cause the gate
        to allow blocked topics without prompting. A developer or CI shell that
        exports BYPASS_TOOL_CONSENT=true (common in agent/automation contexts)
        would otherwise make the no-context, allowlist, and interrupt cases pass
        silently and fail their assertions. Clearing both per-test makes each
        case deterministic regardless of the ambient environment; tests that
        exercise those paths opt in explicitly via monkeypatch.setenv.
        """
        monkeypatch.delenv("BYPASS_TOOL_CONSENT", raising=False)
        monkeypatch.delenv("STRANDS_ROS2_PUBLISH_ALLOW", raising=False)

    def test_non_blocked_topic_passes(self):
        assert _gate_publish("/my_topic", None) is None

    def test_blocked_topic_no_context_returns_error(self):
        result = _gate_publish("/cmd_vel", None)
        assert result is not None
        assert result["status"] == "error"
        assert "approval" in result["content"][0]["text"].lower()

    def test_allowlist_skips_gate(self, monkeypatch):
        monkeypatch.setenv("STRANDS_ROS2_PUBLISH_ALLOW", "/cmd_vel")
        assert _gate_publish("/cmd_vel", None) is None

    def test_allowlist_namespaced(self, monkeypatch):
        monkeypatch.setenv("STRANDS_ROS2_PUBLISH_ALLOW", "/cmd_vel")
        assert _gate_publish("/my_robot/cmd_vel", None) is None

    def test_allowlist_does_not_cover_other_topics(self, monkeypatch):
        monkeypatch.setenv("STRANDS_ROS2_PUBLISH_ALLOW", "/cmd_vel")
        result = _gate_publish("/emergency_stop", None)
        assert result is not None
        assert result["status"] == "error"

    def test_bypass_consent_allows(self, monkeypatch):
        monkeypatch.setenv("BYPASS_TOOL_CONSENT", "true")
        assert _gate_publish("/cmd_vel", None) is None

    def test_interrupt_approved(self):
        ctx = MagicMock()
        ctx.interrupt.return_value = "y"
        assert _gate_publish("/cmd_vel", ctx) is None
        ctx.interrupt.assert_called_once()
        reason = ctx.interrupt.call_args[1]["reason"]
        assert reason["action"] == "publish"
        assert reason["topic"] == "/cmd_vel"

    def test_interrupt_declined(self):
        ctx = MagicMock()
        ctx.interrupt.return_value = "no"
        result = _gate_publish("/cmd_vel", ctx)
        assert result is not None
        assert result["status"] == "error"
        assert "declined" in result["content"][0]["text"]

    def test_interrupt_runtime_error_fails_closed(self):
        ctx = MagicMock()
        ctx.interrupt.side_effect = RuntimeError("no agent loop")
        result = _gate_publish("/cmd_vel", ctx)
        assert result is not None
        assert result["status"] == "error"

    @pytest.mark.parametrize("response", ["y", "Y", "yes", "YES", "approve", "Approved"])
    def test_approve_response_affirmative(self, response):
        assert _approve_response(response) is True

    @pytest.mark.parametrize("response", ["n", "no", "nope", "", 42, None])
    def test_approve_response_negative(self, response):
        assert _approve_response(response) is False
