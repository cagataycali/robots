"""The Reachy DoA turner turns toward a voice once, on evidence, and never on noise.

Every case drives :class:`~strands_robots.drivers.reachy_doa.DoaTurner` with
scripted daemon frames and an injected clock; nothing here opens a socket or
imports a transport. The loop cases drive :class:`DoaLoop` through ``step``
(no thread) and once through a real thread against an in-memory frame source.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any

import pytest

from strands_robots.drivers import reachy_doa
from strands_robots.drivers.reachy_doa import (
    BODY_MAX_DEG,
    HEAD_MAX_DEG,
    MIN_CONSEC,
    RATE_S,
    DoaLoop,
    DoaTurner,
    TurnPlan,
)

LEFT_60 = math.radians(30.0)  # 0 = left, so 30 deg from the left rail = 60 deg left of centre
AHEAD = math.pi / 2


def _frame(
    angle: float, speech: bool = True, *, yaw_deg: float = 0.0, pitch_deg: float = 0.0, body_deg: float = 0.0
) -> dict[str, Any]:
    return {
        "doa": {"angle": angle, "speech_detected": speech},
        "head_pose": {"roll": 0.0, "pitch": math.radians(pitch_deg), "yaw": math.radians(yaw_deg)},
        "body_yaw": math.radians(body_deg),
    }


def _feed(
    turner: DoaTurner, angle: float, n: int = MIN_CONSEC, *, t0: float = 100.0, **frame_kw: Any
) -> TurnPlan | None:
    plan = None
    for i in range(n):
        plan = turner.observe(_frame(angle, **frame_kw), now=t0 + 0.1 * i)
    return plan


class TestTheStreakIsEvidence:
    def test_one_frame_is_not_a_speaker(self) -> None:
        turner = DoaTurner()
        assert turner.observe(_frame(LEFT_60), now=1.0) is None
        assert turner.why == f"listening 1/{MIN_CONSEC}"

    def test_six_agreeing_speech_frames_plan_a_turn(self) -> None:
        turner = DoaTurner()
        plan = _feed(turner, LEFT_60)
        assert isinstance(plan, TurnPlan)
        assert plan.delta_deg == 60.0
        assert turner.turns == 1
        assert turner.why is None

    def test_silence_breaks_the_streak(self) -> None:
        turner = DoaTurner()
        _feed(turner, LEFT_60, n=MIN_CONSEC - 1)
        assert turner.observe(_frame(LEFT_60, speech=False), now=101.0) is None
        assert turner.why == "no speech"
        assert turner.observe(_frame(LEFT_60), now=101.1) is None
        assert turner.why == f"listening 1/{MIN_CONSEC}"

    def test_a_moving_sound_does_not_turn(self) -> None:
        turner = DoaTurner()
        for i, deg in enumerate((30, 60, 90, 120, 150, 40)):
            plan = turner.observe(_frame(math.radians(deg)), now=100.0 + 0.1 * i)
        assert plan is None
        assert turner.why == "sound is moving"

    def test_a_stale_gap_restarts_the_streak(self) -> None:
        turner = DoaTurner()
        _feed(turner, LEFT_60, n=MIN_CONSEC - 1)
        assert turner.observe(_frame(LEFT_60), now=100.0 + 0.1 * (MIN_CONSEC - 1) + reachy_doa.STALE_S + 0.5) is None
        assert turner.why == f"listening 1/{MIN_CONSEC}"

    def test_rail_readings_are_not_bearings(self) -> None:
        turner = DoaTurner()
        for angle in (0.0, math.radians(1.0), math.pi, math.radians(178.5)):
            assert turner.observe(_frame(angle), now=1.0) is None
            assert turner.why == "rail reading"

    def test_already_facing_does_not_twitch(self) -> None:
        turner = DoaTurner()
        assert _feed(turner, AHEAD + math.radians(4.0)) is None
        assert turner.why is not None and turner.why.startswith("already facing")

    @pytest.mark.parametrize("bad", ["nan", None, True, "12", float("inf")])
    def test_a_non_numeric_angle_is_ignored(self, bad: Any) -> None:
        turner = DoaTurner()
        frame = {"doa": {"angle": float(bad) if bad == "nan" else bad, "speech_detected": True}}
        assert turner.observe(frame, now=1.0) is None
        assert turner.last_angle is None

    def test_the_bare_doa_body_is_accepted_too(self) -> None:
        turner = DoaTurner()
        plan = None
        for i in range(MIN_CONSEC):
            plan = turner.observe({"angle": LEFT_60, "speech_detected": True}, now=100.0 + 0.1 * i)
        assert isinstance(plan, TurnPlan)


class TestThePlanSplitsHeadAndBody:
    def test_a_small_bearing_moves_only_the_head(self) -> None:
        plan = _feed(DoaTurner(), math.radians(60.0))  # 30 deg left of centre
        assert plan is not None
        assert plan.yaw == 30.0
        assert plan.body_yaw is None

    def test_a_large_bearing_hands_the_rest_to_the_body(self) -> None:
        plan = _feed(DoaTurner(), math.radians(10.0))  # 80 deg left
        assert plan is not None
        assert plan.yaw == HEAD_MAX_DEG
        assert plan.body_yaw == 35.0

    def test_the_body_never_exceeds_its_ceiling(self) -> None:
        plan = _feed(DoaTurner(), math.radians(10.0), body_deg=50.0)  # 80 deg left, body already at 50
        assert plan is not None
        assert plan.body_yaw == BODY_MAX_DEG

    def test_head_frame_adds_the_delta_to_the_current_yaw(self) -> None:
        plan = _feed(DoaTurner(frame="head"), math.radians(60.0), yaw_deg=10.0)
        assert plan is not None
        assert plan.from_yaw == 10.0
        assert plan.yaw == 40.0

    def test_body_frame_treats_the_delta_as_absolute(self) -> None:
        plan = _feed(DoaTurner(frame="body"), math.radians(60.0), yaw_deg=10.0)
        assert plan is not None
        assert plan.yaw == 30.0

    def test_pitch_is_carried_but_windowed(self) -> None:
        plan = _feed(DoaTurner(), math.radians(60.0), pitch_deg=30.0)
        assert plan is not None
        assert plan.pitch == reachy_doa.PITCH_WINDOW_DEG[1]

    def test_a_mirrored_array_flips_the_sign(self) -> None:
        plan = _feed(DoaTurner(sign=-1), math.radians(60.0))
        assert plan is not None
        assert plan.yaw == -30.0

    def test_an_unknown_frame_name_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="frame must be 'head' or 'body'"):
            DoaTurner(frame="world")


class TestTurnsAreRateLimitedAndVetoable:
    def test_a_second_turn_within_the_rate_window_waits(self) -> None:
        turner = DoaTurner()
        assert _feed(turner, LEFT_60, t0=100.0) is not None
        assert _feed(turner, math.radians(150.0), t0=101.0) is None
        assert turner.why == "rate limit"
        assert _feed(turner, math.radians(150.0), t0=100.0 + RATE_S + 1.0) is not None

    def test_the_windup_guard_refuses_a_repeat_in_the_same_direction(self) -> None:
        turner = DoaTurner()
        assert _feed(turner, LEFT_60, t0=100.0) is not None
        # rate window passed, same bearing still 60 deg left: the sound moved with us
        assert _feed(turner, LEFT_60, t0=100.0 + RATE_S + 0.5) is None
        assert turner.why == "windup guard"
        assert turner.windups == 1

    def test_a_shrunken_bearing_after_a_turn_is_honoured(self) -> None:
        turner = DoaTurner()
        assert _feed(turner, LEFT_60, t0=100.0) is not None
        # now only 20 deg left: less than 60% of the last delta, so not a windup
        assert _feed(turner, math.radians(70.0), t0=100.0 + RATE_S + 0.5) is not None

    def test_a_blocked_callable_vetoes_only_an_armed_turn(self) -> None:
        calls = 0

        def blocked() -> str | None:
            nonlocal calls
            calls += 1
            return "face tracker has a lock"

        turner = DoaTurner()
        for i in range(MIN_CONSEC):
            turner.observe(_frame(LEFT_60), now=100.0 + 0.1 * i, blocked=blocked)
        assert calls == 1  # consulted once, when the streak was complete
        assert turner.why == "face tracker has a lock"
        assert turner.turns == 0

    def test_a_disabled_turner_still_reports_the_bearing(self) -> None:
        turner = DoaTurner(enabled=False)
        assert _feed(turner, LEFT_60) is None
        assert turner.why == "disabled"
        status = turner.status()
        assert status["angle_deg"] == 30.0
        assert status["delta_deg"] == 60.0
        assert status["enabled"] is False


class TestTheLoopSendsWhatTheTurnerPlans:
    def _loop(self, frames: list[dict[str, Any]], look_status: str = "success") -> tuple[DoaLoop, list[dict[str, Any]]]:
        sent: list[dict[str, Any]] = []

        def look(**kwargs: Any) -> dict[str, Any]:
            sent.append(kwargs)
            return {"status": look_status, "content": [{"text": look_status}]}

        it = iter(frames)
        return DoaLoop(lambda: next(it, None), look), sent

    def test_step_sends_one_goto_shaped_call_in_degrees(self) -> None:
        loop, sent = self._loop([])
        for i in range(MIN_CONSEC):
            loop.step(_frame(LEFT_60), now=100.0 + 0.1 * i)
        assert sent == [{"yaw": 45.0, "body_yaw": 15.0, "pitch": 0.0, "roll": 0.0, "duration": reachy_doa.TURN_S}]
        assert loop.last_error is None
        assert loop.last_sent is not None and loop.last_sent["plan"]["delta_deg"] == 60.0

    def test_a_refused_look_is_recorded_not_raised(self) -> None:
        loop, _ = self._loop([], look_status="error")
        for i in range(MIN_CONSEC):
            loop.step(_frame(LEFT_60), now=100.0 + 0.1 * i)
        assert loop.last_error is not None and loop.last_error.startswith("look refused")

    def test_a_raising_look_is_recorded_not_raised(self) -> None:
        def look(**_: Any) -> dict[str, Any]:
            raise RuntimeError("link down")

        loop = DoaLoop(lambda: None, look)
        for i in range(MIN_CONSEC):
            loop.step(_frame(LEFT_60), now=100.0 + 0.1 * i)
        assert loop.last_error == "look: link down"

    def test_the_thread_starts_reads_and_stops(self) -> None:
        frames = [_frame(LEFT_60) for _ in range(MIN_CONSEC)]
        loop, sent = self._loop(frames)
        loop = DoaLoop(loop._read_frame, loop._look, poll_s=0.005)
        loop.start()
        assert loop.running
        deadline = time.monotonic() + 3.0
        while not sent and time.monotonic() < deadline:
            time.sleep(0.01)
        loop.stop()
        assert not loop.running
        assert len(sent) == 1
        assert loop.turner.enabled is False
        assert not any(t.name == "reachy-doa" for t in threading.enumerate())

    def test_start_twice_is_one_thread_and_stop_twice_is_quiet(self) -> None:
        loop = DoaLoop(lambda: None, lambda **_: {"status": "success"}, poll_s=0.005)
        loop.start()
        first = loop._thread
        loop.start()
        assert loop._thread is first
        loop.stop()
        loop.stop()
        assert not loop.running

    def test_a_failing_frame_source_is_counted_and_survived(self) -> None:
        def read() -> dict[str, Any]:
            raise OSError("daemon away")

        loop = DoaLoop(read, lambda **_: {"status": "success"}, poll_s=0.005)
        loop.start()
        deadline = time.monotonic() + 2.0
        while loop.read_errors == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        loop.stop()
        assert loop.read_errors >= 1
        assert loop.last_error is not None and loop.last_error.startswith("read_frame")

    def test_status_carries_the_turner_and_the_loop(self) -> None:
        loop = DoaLoop(lambda: None, lambda **_: {"status": "success"})
        status = loop.status()
        assert status["running"] is False
        assert status["enabled"] is True
        assert set(status["rules"]) >= {"min_consec", "rate_s", "head_max_deg", "body_max_deg"}

    @pytest.mark.parametrize("bad", [0, -1.0, float("nan"), float("inf")])
    def test_a_bad_poll_period_is_refused(self, bad: float) -> None:
        with pytest.raises(ValueError, match="poll_s must be a positive finite number"):
            DoaLoop(lambda: None, lambda **_: {"status": "success"}, poll_s=bad)
