"""A still-life episode is reported BY THE WORKER, not just by the pure module (Q35 part 2).

record_motion.py is unit-tested on its own, and that is exactly the state R5 was in when its
fix was correct in the lib and inert in the UI for a commit. These tests drive the real
RecordWorker with a frozen follower and read the session payload the frontend polls, so
"the judgment exists" and "the operator is told" cannot come apart.

Deterministic: no thread, no hardware, manual ticks, a clock the test advances.
"""

from __future__ import annotations

import numpy as np

from strands_robots.dashboard import record_motion
from strands_robots.dashboard.record_worker import RecordWorker


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


class Backend:
    """A follower whose joints move only if ``moving`` is set.

    ``moving=False`` is the measured failure: the 12V supply is off, so the bus still
    answers position reads from the USB logic rail and the observation looks perfectly
    valid. Cameras keep producing frames too - nothing here is missing, it is just still.
    """

    def __init__(self, *, moving: bool, suffix: str = ".pos") -> None:
        self.camera_keys = ["top"]
        self.moving = moving
        self.suffix = suffix
        self._n = 0

    def leader_action(self) -> dict[str, float]:
        self._n += 1
        return {"shoulder_pan": float(self._n) if self.moving else 12.0}

    def follower_apply(self, action):
        return action  # a write with no torque behind it still "succeeds"

    def follower_observation(self):
        pos = 12.0 + (self._n * 0.5 if self.moving else 0.0)
        return {
            f"shoulder_pan{self.suffix}": pos,
            f"wrist_roll{self.suffix}": 170.0,
            "top": np.zeros((8, 8, 3), dtype=np.uint8),
        }

    def close(self) -> None:
        pass


class Recorder:
    def __init__(self) -> None:
        self.frames = 0

    def add_frame(self, obs, action, task=None) -> None:
        self.frames += 1

    def save_episode(self):
        return {"status": "ok", "episode_index": 0}

    def clear_episode_buffer(self) -> bool:
        return True

    def finalize(self) -> None:
        pass


def make(*, moving: bool, suffix: str = ".pos", tmp_path=None):
    clock, backend = Clock(), Backend(moving=moving, suffix=suffix)
    worker = RecordWorker(
        dataset="cagatay/so101-pick", task="pick up the cube",
        leader="arm-leader", follower="arm-follower",
        target_episodes=3, fps=5, backend=backend,
        recorder_factory=lambda **_: Recorder(),
        thumb_dir=str(tmp_path or "/tmp/rec-still-life-test"),
        clock=clock, autostart_loop=False,
    )
    return worker, clock


def run(worker: RecordWorker, clock: Clock, ticks: int, dt: float = 0.2) -> None:
    for _ in range(ticks):
        clock.t += dt
        worker.tick()


def test_a_frozen_follower_is_reported_in_the_session_the_frontend_polls(tmp_path) -> None:
    worker, clock = make(moving=False, tmp_path=tmp_path)
    worker.start_episode()
    run(worker, clock, 60)  # 12s at 5Hz - past record_motion's window
    s = worker.session()
    assert s["episodes"][-1]["frames"] == 60, "the frames really were written"
    notice = s["motion_notice"]
    assert notice is not None, "60 frames of one pose were recorded and the session said nothing"
    assert notice["still"] is True
    assert "12V" in notice["message"] and "redo" in notice["message"]
    assert "60 frames" in notice["message"], "it must say how much was recorded that way"


def test_a_moving_follower_is_never_flagged(tmp_path) -> None:
    worker, clock = make(moving=True, tmp_path=tmp_path)
    worker.start_episode()
    run(worker, clock, 60)
    assert worker.session()["motion_notice"] is None


def test_recording_is_not_stopped_or_refused_by_the_notice(tmp_path) -> None:
    # A notice, not a guard: holding still is legitimate, and throwing away a real
    # episode to prevent a suspicion is the more expensive mistake.
    worker, clock = make(moving=False, tmp_path=tmp_path)
    worker.start_episode()
    run(worker, clock, 60)
    assert worker.session()["phase"] == "recording"
    assert worker.session()["error"] is None
    kept = worker.stop_episode()
    assert kept["episodes"][-1]["frames"] == 60


def test_the_notice_survives_stop_so_it_can_be_read_between_episodes(tmp_path) -> None:
    worker, clock = make(moving=False, tmp_path=tmp_path)
    worker.start_episode()
    run(worker, clock, 60)
    worker.stop_episode()
    assert worker.session()["motion_notice"] is not None, (
        "the operator reads this panel after pressing stop; a notice that vanishes then "
        "is only ever seen by someone watching the screen at the time"
    )


def test_a_new_episode_starts_with_a_clean_slate(tmp_path) -> None:
    # The gap between episodes is when the operator lines the arms up by hand, and that
    # pause must not be attributed to the next episode.
    worker, clock = make(moving=False, tmp_path=tmp_path)
    worker.start_episode()
    run(worker, clock, 60)
    worker.stop_episode()
    clock.t += 30.0  # a long think between episodes
    worker.start_episode()
    assert worker.session()["motion_notice"] is None
    run(worker, clock, 5)
    assert worker.session()["motion_notice"] is None, "5 frames is not evidence of anything"


def test_stillness_while_merely_paused_is_not_recorded_as_stillness(tmp_path) -> None:
    # tick() teleops in every phase but records only while ``recording``. A frozen arm
    # that nobody is recording is not this notice's business: no dataset is at risk.
    worker, clock = make(moving=False, tmp_path=tmp_path)
    run(worker, clock, 60)  # never started an episode
    s = worker.session()
    assert s["episodes"] == []
    assert s["motion_notice"] is None


def test_an_observation_schema_without_pos_keys_stays_silent(tmp_path) -> None:
    # Silence beats guessing: a backend whose joints are not named ``*.pos`` (a sim, or a
    # future schema) must not be reported as a frozen arm on the strength of a rule that
    # never found the joints in the first place.
    worker, clock = make(moving=False, suffix="", tmp_path=tmp_path)
    worker.start_episode()
    run(worker, clock, 60)
    assert worker.session()["motion_notice"] is None


def test_the_sample_ring_stays_bounded_by_the_window(tmp_path) -> None:
    # 200s of recording must not accumulate 1000 samples: this runs per frame at up to
    # 30fps for as long as an operator keeps collecting.
    worker, clock = make(moving=False, tmp_path=tmp_path)
    worker.start_episode()
    run(worker, clock, 1000)
    ring = worker._motion  # noqa: SLF001 - the bound is the point of the test
    assert len(ring) <= record_motion.WINDOW_S * 2 * 5 + 2, len(ring)
    assert ring[-1][0] - ring[0][0] <= record_motion.WINDOW_S * 2 + 0.001
