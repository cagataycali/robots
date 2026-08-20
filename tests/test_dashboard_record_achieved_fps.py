"""A record session must report the rate it CAPTURED, not only the one it declares.

BUGS.md Q70. A LeRobotDataset timestamps every frame positionally as
``frame_index / fps``, so the declared rate is written into the artifact and the
captured rate is written nowhere - no wall-clock column exists that could ever
contradict it (verified on a real dataset: every delta exactly 0.1000s at
fps=10). Before this, ``session()`` reported ``fps`` - the DECLARATION - and the
dashboard's own control loop was paced by an ``Event.wait`` inflated ~137ms in a
daemon-descended tree, so a session opened at 30 fps stepped at ~5.6Hz and every
surface said success. The mislabel then reaches a policy as a control period 5x
longer than the one it is told about, and ``replay_episode`` moves at the wrong
speed for the same reason.

So the pair of numbers has to be visible while the operator can still act on it.
This file pins that the measurement is honest about what it cannot know, that
the notice fires on a real gap and stays silent on jitter, and that it NEVER
becomes a refusal: the operator is holding a leader arm, and aborting their
episode over a rate they cannot change from that position destroys work to
prevent a mislabel.
"""

from strands_robots.dashboard.record_worker import (
    FPS_DRIFT_TOLERANCE,
    achieved_fps,
    fps_verdict,
)

from tests.test_dashboard_record_worker import make_worker


class TestWhatCannotBeMeasuredIsNotGuessed:
    def test_one_frame_has_no_rate(self):
        """One timestamp spans no interval. Reporting 1.0 would be a fiction."""
        assert achieved_fps(1, 0.5) is None
        assert achieved_fps(0, 0.0) is None

    def test_two_frames_are_enough(self):
        """Early truth beats late precision: a 5x error shows in two frames."""
        assert achieved_fps(2, 0.2) == 10.0

    def test_a_zero_or_negative_duration_is_refused(self):
        assert achieved_fps(10, 0.0) is None
        assert achieved_fps(10, -1.0) is None

    def test_the_window_starts_when_the_EPISODE_did_not_at_the_first_frame(self):
        """The episode opens one period BEFORE frame 1 (the loop waits, then ticks),
        so the window already holds one interval per frame. Dividing by frames-1
        here understates a short episode: 6 frames over 1.2s of a real 5Hz loop
        would read 4.17. Measured against the loop, this is the honest form."""
        assert achieved_fps(30, 1.0) == 30.0
        assert achieved_fps(6, 1.2) == 5.0


class TestTheNoticeSpeaksOnlyWhenItMatters:
    def test_no_measurement_is_silence(self):
        assert fps_verdict(30, None) is None

    def test_jitter_inside_the_tolerance_is_silence(self):
        """A warning that fires on healthy scheduling teaches operators to ignore it."""
        assert fps_verdict(30, 30.0) is None
        assert fps_verdict(30, 29.0) is None
        assert fps_verdict(30, 30 * (1 + FPS_DRIFT_TOLERANCE)) is None

    def test_the_q70_case_names_the_gap_and_its_consequence(self):
        v = fps_verdict(30, 5.6)
        assert v is not None
        assert v["declared_fps"] == 30 and v["measured_fps"] == 5.6
        assert v["slower"] is True
        assert v["ratio"] == 5.36
        assert "closer together than reality" in v["detail"]
        assert "wrong control period" in v["detail"], "the consequence, not the metric"
        assert "replay" in v["detail"]

    def test_it_says_what_to_do_and_the_advice_is_reachable(self):
        """'Re-open at N fps' has to name an N the loop demonstrably achieves."""
        v = fps_verdict(30, 5.6)
        assert v is not None and "re-open the session at 5 fps" in v["detail"]

    def test_faster_than_declared_is_also_wrong_and_says_so(self):
        """The inverse mislabel is just as real: timestamps stretched, not squeezed."""
        v = fps_verdict(10, 30.0)
        assert v is not None
        assert v["slower"] is False and v["ratio"] == 3.0
        assert "further apart than reality" in v["detail"]

    def test_it_is_a_notice_and_never_an_exception(self):
        """Pinned deliberately: a mid-session refusal costs the operator the episode."""
        for measured in (0.001, 5.6, 1000.0):
            assert isinstance(fps_verdict(30, measured), dict)


class TestTheSessionCarriesBothNumbers:
    """The declared/captured pair must reach the screen that collects."""

    def _record(self, w, clock, *, frames, step):
        w.start_episode()
        for _ in range(frames):
            clock.t += step
            w.tick()

    def test_a_fresh_session_measures_nothing_but_still_declares(self):
        w, _, _, _ = make_worker()
        s = w.session()
        assert s["fps"] == 30, "the declaration is unchanged - it is the wire contract"
        assert s["fps_achieved"] is None
        assert s["fps_notice"] is None

    def test_a_slow_loop_is_reported_while_the_episode_is_still_open(self):
        """Mid-episode, not at close: the operator can still stop and re-open."""
        w, _, _, clock = make_worker()
        self._record(w, clock, frames=6, step=1 / 5.0)  # 5Hz against a declared 30
        s = w.session()
        assert s["fps_achieved"] == 5.0
        assert s["fps_notice"] is not None
        assert s["fps_notice"]["ratio"] == 6.0
        assert s["episodes"][-1]["fps_achieved"] == 5.0, "per-episode truth too"

    def test_a_healthy_loop_reports_the_rate_and_no_notice(self):
        w, _, _, clock = make_worker()
        self._record(w, clock, frames=10, step=1 / 30.0)
        s = w.session()
        assert s["fps_achieved"] == 30.0
        assert s["fps_notice"] is None

    def test_the_rate_survives_between_episodes(self):
        """Paused is exactly when the panel is read; the number must not vanish."""
        w, _, _, clock = make_worker()
        self._record(w, clock, frames=6, step=1 / 5.0)
        w.stop_episode()
        s = w.session()
        assert s["fps_achieved"] == 5.0
        assert s["fps_notice"] is not None

    def test_a_discarded_episode_does_not_speak_for_the_dataset(self):
        """Its frames were thrown away, so its rate describes nothing trainable.

        ``discard`` is the verb here, not ``redo``: redo only drops the buffer of
        an episode still recording, so after ``stop_episode`` it is a no-op - as
        this test discovered by asserting None and being handed 5.0.
        """
        w, _, _, clock = make_worker()
        self._record(w, clock, frames=6, step=1 / 5.0)
        w.stop_episode()
        w.discard(w.session()["episodes"][-1]["index"])
        s = w.session()
        assert s["fps_achieved"] is None
        assert s["fps_notice"] is None
