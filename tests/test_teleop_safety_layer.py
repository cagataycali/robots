# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""The teleop loop needs a clamp, a rate limit and a stale-leader watchdog.

Two real sub-defects (the other two in the original finding were refuted by the
ledger's verifier and are pinned as negative tests at the bottom):

1. ``_teleop_loop`` passed the merged leader action straight to ``send_action``
   with no joint-limit check and no per-tick delta cap. The ``joint_limits=``
   kwarg the host class accepts was threaded ONLY into the ROS 2 bridges -
   ``grep joint_limits teleop_mixin.py`` returned nothing - so the same robot
   enforced its documented limits on a ROS command but not on a physically
   attached leader arm. The only remaining protection,
   ``max_relative_target``, is not declared at all by whole robot families
   (``bi_so_follower``, ``bi_openarm_follower``, ``lekiwi_client``,
   ``unitree_g1``, ...), where it is silently dropped.

2. No stale-frame watchdog: a wedged leader USB link keeps ``is_connected``
   True and ``get_action()`` returning the last pose forever. Measured:
   ``frames=50, errors=0, status='success'`` for a session in which exactly one
   pose was applied.

Both new behaviours are opt-in (the kwargs default ``None``), so existing
callers are unchanged.

No serial port is opened and no arm is commanded.
"""

from __future__ import annotations

import logging

import pytest

from strands_robots.teleop_mixin import TeleopMixin, _clamp_teleop_action

_KEYS = ("shoulder_pan.pos", "shoulder_lift.pos")
_LIMITS = {key: (-90.0, 90.0) for key in _KEYS}


class _Leader:
    """A teleoperator returning a scripted sequence of frames."""

    def __init__(self, frames: list[dict[str, float]]) -> None:
        self._frames = frames
        self.calls = 0
        self.is_connected = False

    def connect(self) -> None:
        self.is_connected = True

    def disconnect(self) -> None:
        self.is_connected = False

    def get_action(self) -> dict[str, float]:
        frame = self._frames[min(self.calls, len(self._frames) - 1)]
        self.calls += 1
        return dict(frame)


class _Host(TeleopMixin):
    """Minimal TeleopMixin host: records what reached send_action."""

    def __init__(self) -> None:
        self.applied: list[dict[str, float]] = []

    def send_action(self, action, robot_name=None):
        self.applied.append(dict(action))
        return {"status": "success", "content": [{"text": "ok"}]}


def _drive(frames: list[dict[str, float]], *, ticks: int = 6, **kwargs) -> tuple[_Host, dict]:
    """Run a bounded blocking teleop session and return (host, result).

    Bounded by a tiny duration at a high hz so the loop runs roughly ``ticks``
    iterations without any sleep-based flakiness in the assertions.
    """
    host = _Host()
    host.attach_teleop(_Leader(frames), name="leader")
    result = host.teleoperate(block=True, hz=1000.0, duration=ticks / 1000.0, **kwargs)
    return host, result


class TestClampHelper:
    def test_in_range_frame_passes_unchanged(self):
        action = {"a": 5.0, "b": -5.0}
        assert _clamp_teleop_action(action, {}, {"a": (-10.0, 10.0)}, None) == (action, "")

    def test_out_of_range_rejects_the_WHOLE_frame(self):
        """Reject-whole matches the ROS bridge; a partial pose is a different pose."""
        _, reason = _clamp_teleop_action({"a": 50.0, "b": -5.0}, {}, {"a": (-10.0, 10.0)}, None)

        assert "a=50" in reason
        assert "outside" in reason

    def test_a_key_with_no_declared_limit_is_allowed(self):
        assert _clamp_teleop_action({"z": 999.0}, {}, {"a": (-1.0, 1.0)}, None)[1] == ""

    def test_delta_cap_limits_the_step(self):
        limited, reason = _clamp_teleop_action({"a": 10.0}, {"a": 0.0}, None, 2.0)

        assert reason == ""
        assert limited == {"a": 2.0}

    def test_delta_cap_is_symmetric(self):
        assert _clamp_teleop_action({"a": -10.0}, {"a": 0.0}, None, 2.0)[0] == {"a": -2.0}

    def test_a_small_step_is_untouched(self):
        assert _clamp_teleop_action({"a": 1.0}, {"a": 0.0}, None, 2.0)[0] == {"a": 1.0}

    def test_first_tick_has_no_previous_pose_to_cap_against(self):
        assert _clamp_teleop_action({"a": 99.0}, {}, None, 2.0)[0] == {"a": 99.0}

    def test_no_config_is_a_passthrough(self):
        action = {"a": 1234.0}
        assert _clamp_teleop_action(action, {"a": 0.0}, None, None) == (action, "")


class TestJointLimitsInTheLoop:
    def test_an_out_of_range_frame_is_never_sent(self):
        """Regression: this reached send_action with no check at all."""
        host, result = _drive([{_KEYS[0]: 500.0}], joint_limits=_LIMITS)

        assert host.applied == [], "an out-of-range leader frame reached the arm"
        assert result["status"] != "success"

    def test_an_in_range_frame_is_sent(self):
        host, _ = _drive([{_KEYS[0]: 10.0}], joint_limits=_LIMITS)

        assert host.applied, "a valid frame was blocked"
        assert host.applied[0][_KEYS[0]] == 10.0

    def test_the_rejection_is_logged_once_with_the_reason(self, caplog):
        with caplog.at_level(logging.WARNING):
            _drive([{_KEYS[0]: 500.0}], ticks=12, joint_limits=_LIMITS)

        rejections = [r.getMessage() for r in caplog.records if "REJECTED" in r.getMessage()]
        assert len(rejections) == 1, rejections
        assert "outside" in rejections[0]

    def test_without_joint_limits_nothing_is_gated(self):
        """Opt-in: existing callers must be unaffected."""
        host, _ = _drive([{_KEYS[0]: 500.0}])

        assert host.applied, "the frame was gated without joint_limits configured"


class TestMaxStepInTheLoop:
    def test_a_jump_is_clamped_not_rejected(self):
        frames = [{_KEYS[0]: 0.0}, {_KEYS[0]: 80.0}]
        host, _ = _drive(frames, ticks=8, max_step=5.0)

        assert len(host.applied) >= 2
        # The first frame establishes the pose; the jump is rate-limited.
        assert host.applied[1][_KEYS[0]] == pytest.approx(5.0)

    def test_the_pose_converges_over_ticks(self):
        frames = [{_KEYS[0]: 0.0}, {_KEYS[0]: 80.0}]
        host, _ = _drive(frames, ticks=20, max_step=5.0)

        applied = [a[_KEYS[0]] for a in host.applied]
        assert applied[-1] > applied[1], f"the cap did not converge: {applied}"


class TestStaleLeaderWatchdog:
    def test_a_frozen_leader_is_not_reported_as_success(self):
        """Regression: frames=50, errors=0, status='success', one pose applied."""
        host, result = _drive([{_KEYS[0]: 1.0}], ticks=40, stale_ticks=3)

        assert result["status"] != "success", result
        # It stopped early rather than replaying forever.
        assert len(host.applied) <= 5, len(host.applied)

    def test_the_stop_reason_names_the_leader(self, caplog):
        with caplog.at_level(logging.ERROR):
            _drive([{_KEYS[0]: 1.0}], ticks=40, stale_ticks=3)

        messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
        assert any("unchanged" in m and "stale" in m for m in messages), messages
        assert any("USB" in m for m in messages), messages

    def test_a_moving_leader_is_not_flagged(self):
        frames = [{_KEYS[0]: float(i)} for i in range(30)]
        host, result = _drive(frames, ticks=20, stale_ticks=3)

        assert result["status"] == "success", result
        assert len(host.applied) > 5

    def test_without_stale_ticks_the_watchdog_is_off(self):
        """Opt-in: the previous replay-forever behaviour is preserved by default."""
        host, result = _drive([{_KEYS[0]: 1.0}], ticks=15)

        assert result["status"] == "success"
        assert len(host.applied) > 5


class TestRefutedClaimsArePinned:
    def test_stop_teleoperate_disconnects_all_devices_by_design(self):
        """The 'block=True over-disconnects' claim was REFUTED; pin the design.

        ``stop_teleoperate`` itself iterates ALL of ``self._teleops``, so
        ``block=True``'s teardown is consistent with it rather than a bug. This
        retires KNOWLEDGE_GRAPH B14.
        """
        host = _Host()
        first, second = _Leader([{_KEYS[0]: 0.0}]), _Leader([{_KEYS[1]: 0.0}])
        host.attach_teleop(first, name="a")
        host.attach_teleop(second, name="b")
        host.teleoperate(names=["a"], block=True, hz=1000.0, duration=0.003)

        host.stop_teleoperate()

        assert not first.is_connected
        assert not second.is_connected
