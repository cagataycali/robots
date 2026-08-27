"""A recorded episode that never moves must SAY so (BUGS.md Q35).

The defect being pinned: a follower whose 12V supply trips keeps answering position reads
from the logic rail, so frames land at full fps with valid numbers and the dataset is a
still life. Every assertion here is about what the operator is told, and about the two
directions this check must not fail in - never calling a moving arm frozen, and never
staying silent about a frozen one.
"""

from __future__ import annotations

import math

from strands_robots.dashboard import record_motion as rm


def _still(n: int, *, start: float = 100.0, step: float = 0.2, jitter: float = 0.1) -> list:
    """``n`` samples of a POWERED but stationary arm: sensor jitter only."""
    return [
        (start + i * step, {"shoulder_pan.pos": 12.0 + (jitter if i % 2 else 0.0), "wrist_roll.pos": 170.0})
        for i in range(n)
    ]


class TestNothingIsSaidWithoutEvidence:
    """``None`` is "cannot say yet" - it must never be produced by a frozen arm."""

    def test_a_fresh_episode_says_nothing(self) -> None:
        assert rm.motion_verdict(_still(3), now=100.6) is None

    def test_too_few_samples_says_nothing_even_across_a_long_window(self) -> None:
        # Two identical samples 30s apart are not evidence: a stream that slow is its own
        # problem, and calling it frozen would blame the arm for a delivery failure.
        samples = [(100.0, {"a.pos": 1.0}), (130.0, {"a.pos": 1.0})]
        assert rm.motion_verdict(samples, now=130.0) is None

    def test_a_window_not_yet_covered_says_nothing(self) -> None:
        # 20 samples, but they only span 3.8s - a hold that short is ordinary.
        assert rm.motion_verdict(_still(20), now=103.8) is None

    def test_observations_without_joint_positions_say_nothing(self) -> None:
        samples = [(100.0 + i * 0.2, {"top": object()}) for i in range(60)]  # camera-only
        assert rm.motion_verdict(samples, now=112.0) is None  # type: ignore[arg-type]

    def test_no_samples_at_all_says_nothing(self) -> None:
        assert rm.motion_verdict([], now=100.0) is None


class TestAMovingArmIsNeverCalledFrozen:
    """The expensive direction of error: refusing to trust a real episode."""

    def test_hand_guided_motion_is_motion(self) -> None:
        samples = [(100.0 + i * 0.2, {"shoulder_pan.pos": 12.0 + i * 0.7}) for i in range(60)]
        assert rm.motion_verdict(samples, now=112.0) is None

    def test_a_swing_that_returns_to_the_same_pose_is_motion(self) -> None:
        # first-vs-last would read 0 travel here. Peak-to-peak is why it does not.
        samples = [(100.0 + i * 0.2, {"shoulder_pan.pos": 12.0 + 20.0 * math.sin(i / 10.0)}) for i in range(60)]
        assert rm.motion_verdict(samples, now=112.0) is None

    def test_one_joint_moving_is_enough(self) -> None:
        samples = [(100.0 + i * 0.2, {"a.pos": 5.0, "b.pos": 5.0, "gripper.pos": 5.0 + i * 0.3}) for i in range(60)]
        assert rm.motion_verdict(samples, now=112.0) is None

    def test_motion_just_over_the_epsilon_counts(self) -> None:
        # Travel is judged over the WINDOW, not over the episode: 0.014 deg per sample at
        # 5Hz is 0.55 deg across the last 8s, just past the threshold. (Writing this test
        # against the whole sample list first is what taught me the difference.)
        samples = [(100.0 + i * 0.2, {"a.pos": 5.0 + i * 0.014}) for i in range(60)]
        assert rm.motion_verdict(samples, now=112.0) is None

    def test_motion_just_under_the_epsilon_is_reported(self) -> None:
        # The other side of the same boundary, so the threshold is pinned from both ends.
        samples = [(100.0 + i * 0.2, {"a.pos": 5.0 + i * 0.002}) for i in range(60)]
        assert rm.motion_verdict(samples, now=112.0) is not None


class TestAFrozenArmIsReportedWithItsConsequence:
    def test_a_still_arm_over_the_window_is_reported(self) -> None:
        v = rm.motion_verdict(_still(60), now=112.0, frames=354)
        assert v is not None, "hundreds of identical frames were recorded and nothing said so"
        assert v["still"] is True
        assert v["seconds"] >= rm.WINDOW_S
        assert v["max_travel_deg"] <= rm.EPSILON_DEG

    def test_the_message_names_the_physical_cause_and_the_next_action(self) -> None:
        v = rm.motion_verdict(_still(60), now=112.0, frames=354)
        assert v is not None
        msg = v["message"]
        assert "354 frames" in msg, "the operator needs to know how much was recorded"
        assert "logic rail" in msg and "12V" in msg, "the cause that makes this look healthy"
        assert "redo" in msg, "and what to do about it"
        assert "deliberate" in msg, "a hold is legitimate - this is a notice, not an accusation"

    def test_it_is_a_notice_not_a_refusal(self) -> None:
        # The whole contract: a dict, no exception, nothing that could stop a session.
        assert isinstance(rm.motion_verdict(_still(60), now=112.0), dict)


class TestCoverageComesFromHistoryNotFromThePrunedWindow:
    """The bug that bit twice: a window pruned at ``now - window_s`` has a span about one
    sample interval SHORT of the window at any fps, so judging coverage on it is silent
    about a frozen arm forever. Coverage is therefore read off the whole history."""

    def test_samples_on_a_grid_that_never_lands_exactly_on_the_cutoff(self) -> None:
        # 5Hz starting at 100.2: pruning at 104.0 keeps 104.2 onwards - 7.8s, not 8.0.
        samples = [(100.2 + i * 0.2, {"a.pos": 12.0}) for i in range(60)]
        assert rm.motion_verdict(samples, now=112.0) is not None

    def test_history_shorter_than_the_window_still_says_nothing(self) -> None:
        samples = [(100.2 + i * 0.2, {"a.pos": 12.0}) for i in range(30)]
        assert rm.motion_verdict(samples, now=106.2) is None


class TestAStalledStreamIsNotAFrozenArm:
    """Two different defects; sending the operator to the wrong one wastes the episode."""

    def test_samples_that_stopped_arriving_say_nothing(self) -> None:
        # 60 still samples, then 9s of nothing: no frames are being recorded at all, so
        # this is a capture failure, not a dead power supply.
        assert rm.motion_verdict(_still(60), now=121.0) is None

    def test_a_fresh_still_stream_is_still_reported(self) -> None:
        # The boundary of the rule above: samples up to ~now, and it speaks.
        assert rm.motion_verdict(_still(60), now=112.0) is not None


class TestTheRingIsBoundedByTimeNotByCount:
    def test_old_samples_are_dropped(self) -> None:
        samples = _still(120)  # 24s at 5Hz
        kept = rm.prune(samples, now=samples[-1][0], window_s=8.0)
        assert kept and kept[0][0] >= samples[-1][0] - 8.0
        assert len(kept) < len(samples)

    def test_pruning_to_the_window_is_what_the_verdict_measures(self) -> None:
        # 60s of stillness followed by real motion in the last 8s: NOT frozen, because the
        # verdict must describe the arm NOW, not the arm's whole history.
        still = [(100.0 + i * 0.2, {"a.pos": 5.0}) for i in range(300)]
        moving = [(160.0 + i * 0.2, {"a.pos": 5.0 + i * 0.5}) for i in range(60)]
        assert rm.motion_verdict(still + moving, now=172.0) is None


class TestWhichNumbersCount:
    def test_only_pos_keys_are_read(self) -> None:
        got = rm.joint_positions({"a.pos": 1.5, "a.vel": 99.0, "top": object(), "task": "pick"})
        assert got == {"a.pos": 1.5}

    def test_non_finite_positions_are_dropped_not_kept_as_nan(self) -> None:
        # NaN compares false against everything, so keeping it would make a FROZEN arm
        # measure as moving - the one direction of error that hides the defect.
        got = rm.joint_positions({"a.pos": float("nan"), "b.pos": float("inf"), "c.pos": 3.0})
        assert got == {"c.pos": 3.0}

    def test_booleans_are_not_positions(self) -> None:
        assert rm.joint_positions({"a.pos": True, "b.pos": 2.0}) == {"b.pos": 2.0}

    def test_a_missing_observation_is_empty_not_an_error(self) -> None:
        assert rm.joint_positions(None) == {}

    def test_a_joint_that_appears_late_does_not_read_as_frozen(self) -> None:
        # Peak-to-peak over the joints PRESENT in each sample: a joint that only appears
        # in the last few samples must not contribute a fake 0-travel verdict on its own.
        samples = [(100.0 + i * 0.2, {"a.pos": 5.0 + i * 0.5}) for i in range(60)]
        samples[-1][1]["late.pos"] = 1.0
        assert rm.motion_verdict(samples, now=112.0) is None
