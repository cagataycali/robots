"""Recording surface of the operator dashboard (consolidated).

Consolidated verbatim from: test_dashboard_record_achieved_fps.py, test_dashboard_record_camera_notice.py, test_dashboard_record_crash.py, test_dashboard_record_fake_fidelity.py, test_dashboard_record_fps_declared.py, test_dashboard_record_joint_gate.py, test_dashboard_record_motion.py, test_dashboard_record_still_life.py, test_dashboard_record_target_name.py, test_dashboard_record_upload_honesty.py.
Each section keeps its original tests unchanged.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import pathlib
import tempfile

import numpy as np
import pytest

from strands_robots.dashboard import record_crash, record_joints, record_motion
from strands_robots.dashboard import record_motion as rm
from strands_robots.dashboard.dataset_check import record_target_verdict
from strands_robots.dashboard.record_worker import (
    FPS_DRIFT_TOLERANCE,
    RecordWorker,
    achieved_fps,
    camera_verdict,
    fps_verdict,
    upload_verdict,
)
from strands_robots.dataset_recorder import DatasetRecorder
from tests.test_dashboard_record_api import FakeRecorder as ApiFake
from tests.test_dashboard_record_worker import FakeRecorder as WorkerFake
from tests.test_dashboard_record_worker import make_worker

# ============================================================================
# from tests/test_dashboard_record_achieved_fps.py
# A record session must report the rate it CAPTURED, not only the one it declares.
# ============================================================================


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


# ============================================================================
# from tests/test_dashboard_record_camera_notice.py
# A requested camera that never opened must be visible BEFORE collection.
# ============================================================================


class TestSilenceWhenThereIsNothingToSay:
    def test_all_requested_cameras_present_is_silent(self):
        assert camera_verdict({"top": {}, "wrist": {}}, ["top", "wrist"]) is None

    def test_no_cameras_requested_and_none_present_is_silent(self):
        assert camera_verdict({}, []) is None
        assert camera_verdict(None, None) is None

    def test_an_extra_present_camera_is_not_a_problem(self):
        """Only the requested set is a promise; a bonus channel breaks nothing."""
        assert camera_verdict({"top": {}}, ["top", "wrist"]) is None


class TestTheVerdictNamesTheConsequence:
    def test_every_camera_missing_says_the_dataset_has_no_image_channel(self):
        v = camera_verdict({"top": {}, "wrist": {}}, [])
        assert v is not None
        assert v["missing"] == ["top", "wrist"]
        assert v["present"] == []
        assert "NO image channel" in v["message"]
        assert "cannot train a visual policy" in v["message"]

    def test_one_camera_missing_names_only_that_one(self):
        v = camera_verdict({"top": {}, "wrist": {}}, ["top"])
        assert v is not None
        assert v["missing"] == ["wrist"]
        assert v["present"] == ["top"]
        assert "wrist" in v["message"] and "missing those image channels" in v["message"]
        assert "NO image channel" not in v["message"], "one live camera is not zero"

    def test_the_message_points_at_where_the_reason_lives(self):
        v = camera_verdict({"top": {}}, [])
        assert v is not None
        assert "log" in v["message"], "the operator needs somewhere to look"
        assert "daemon" in v["message"], "the macOS TCC trap is the common cause here"

    def test_counts_are_requested_not_missing(self):
        v = camera_verdict({"a": {}, "b": {}, "c": {}}, ["a"])
        assert v is not None
        assert "2 of 3" in v["message"]


class TestTheSessionCarriesIt:
    """The verdict is only useful if the screen that collects can see it."""

    def test_session_exposes_the_notice_and_a_plain_backend_stays_none(self):
        from tests.test_dashboard_record_worker import make_worker

        worker, backend, _rec, _clock = make_worker()
        session = worker.session()
        assert "camera_notice" in session, "the record screen cannot warn about what it is not told"
        assert session["camera_notice"] is None, "a backend without the attribute must not break"

        # A backend that DID measure a missing camera surfaces it unchanged.
        backend.camera_notice = camera_verdict({"top": {}, "wrist": {}}, ["top"])
        assert worker.session()["camera_notice"]["missing"] == ["wrist"]


class TestTheFinishedDatasetSaysItToo:
    """A receipt that omits the defect sends the operator to training to find it."""

    def _worker_with(self, notice):
        from tests.test_dashboard_record_worker import make_worker

        worker, backend, recorder, _clock = make_worker()
        backend.camera_notice = notice
        return worker, recorder

    def test_close_detail_names_the_missing_cameras_and_the_consequence(self):
        worker, _rec = self._worker_with(camera_verdict({"top": {}, "wrist": {}}, []))
        result = worker.close()
        assert result["ok"] is True
        assert "top, wrist" in result["detail"]
        assert "cannot train a visual policy" in result["detail"]
        assert result["camera_notice"]["missing"] == ["top", "wrist"]

    def test_a_partial_loss_is_not_reported_as_total(self):
        worker, _rec = self._worker_with(camera_verdict({"top": {}, "wrist": {}}, ["top"]))
        detail = worker.close()["detail"]
        assert "wrist" in detail
        assert "cannot train a visual policy" not in detail, "one live camera is not zero"

    def test_a_healthy_session_keeps_its_plain_receipt(self):
        worker, _rec = self._worker_with(None)
        result = worker.close()
        assert "camera_notice" not in result
        assert "WITHOUT" not in result["detail"]
        assert result["detail"].endswith("episode(s) kept")


# ============================================================================
# from tests/test_dashboard_record_crash.py
# A recording the dashboard died inside is not silence (Q40).
# ============================================================================


def test_a_closed_session_leaves_no_trace(tmp_path) -> None:
    p = tmp_path / "crumb.json"
    record_crash.write_crumb({"dataset": "local/x", "leader": "a", "follower": "b"}, path=p, now=1000.0)
    assert record_crash.read_crumb(p) is not None
    record_crash.clear_crumb(p)
    assert record_crash.read_crumb(p) is None
    # Clearing twice is not an error: close() runs it in a finally block.
    record_crash.clear_crumb(p)


def test_the_notice_names_the_dataset_the_arms_and_the_age(tmp_path) -> None:
    p = tmp_path / "crumb.json"
    record_crash.write_crumb(
        {"dataset": "local/cubes", "task": "pick the cube", "leader": "so101-arm-1", "follower": "so101-arm-2"},
        path=p,
        now=1000.0,
    )
    n = record_crash.interrupted_notice(record_crash.read_crumb(p), now=1000.0 + 3 * 60)
    assert n is not None
    assert n["dataset"] == "local/cubes"
    assert "about 3 minutes ago" in n["text"]
    assert "so101-arm-1 and so101-arm-2" in n["text"]
    assert "left despawned" in n["text"], "the arms are still parked - that is the actionable part"
    # It must not claim the dataset is broken, and must not pretend the in-flight episode survived.
    assert "not flushed" in n["text"]
    assert "corrupt" not in n["text"].lower()
    # Both real next actions are NAMED, never performed.
    assert any("delete" in s for s in n["next"])
    assert any("name is taken" in s for s in n["next"])


def test_a_crumb_from_this_very_process_is_not_called_a_restart(tmp_path) -> None:
    # A crumb written by THIS pid with no live worker means the session ended without closing
    # inside a running dashboard - a different fault, and "the dashboard stopped" would be a
    # confident invention.
    crumb = {"dataset": "local/x", "opened_at": 1000.0, "pid": os.getpid()}
    same = record_crash.interrupted_notice(crumb, now=1000.0, same_process=True)
    other = record_crash.interrupted_notice(crumb, now=1000.0, same_process=False)
    assert same is not None and other is not None
    assert "opened and never closed" in same["text"]
    assert "dashboard stopped" in other["text"]


def test_no_evidence_produces_no_notice(tmp_path) -> None:
    assert record_crash.interrupted_notice(None) is None
    assert record_crash.interrupted_notice({}) is None
    assert record_crash.interrupted_notice({"dataset": "  "}) is None
    # A corrupt or half-written crumb is no evidence, not an error.
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert record_crash.read_crumb(bad) is None
    bad.write_text(json.dumps({"task": "no dataset here"}))
    assert record_crash.read_crumb(bad) is None
    assert record_crash.read_crumb(tmp_path / "absent.json") is None


def test_an_unwritable_home_does_not_stop_a_recording(tmp_path) -> None:
    # write_crumb is a courtesy; a read-only home must never be why a session refuses to open.
    record_crash.write_crumb({"dataset": "local/x"}, path=tmp_path / "no" / "such" / "\0bad")
    record_crash.clear_crumb(tmp_path / "nope" / "gone.json")


def test_an_unknown_open_time_says_so(tmp_path) -> None:
    n = record_crash.interrupted_notice({"dataset": "local/x"}, now=1000.0)
    assert n is not None and "at an unknown time" in n["text"]
    assert n["opened_ago"] is None


def test_the_controller_reports_it_when_idle(tmp_path, monkeypatch) -> None:
    from strands_robots.dashboard import record_api

    crumb = tmp_path / "crumb.json"
    monkeypatch.setenv("STRANDS_DASH_RECORD_CRUMB", str(crumb))
    record_crash.write_crumb({"dataset": "local/cubes", "leader": "a", "follower": "b"}, path=crumb, now=1.0)

    ctl = record_api.RecordController(devices=object(), backend_factory=lambda **_: object())
    idle = ctl.session()
    # The idle shape is unchanged for every existing client...
    assert idle["dataset"] is None and idle["phase"] == "idle"
    # ...and the evidence rides alongside it.
    assert idle["interrupted"]["dataset"] == "local/cubes"

    record_crash.clear_crumb(crumb)
    assert (
        "interrupted"
        not in record_api.RecordController(devices=object(), backend_factory=lambda **_: object()).session()
    )


# ============================================================================
# from tests/test_dashboard_record_fake_fidelity.py
# A fake's signature is a CLAIM about the real class -- this pins the claim (Q56 follow-up).
# ============================================================================


def _params(fn) -> tuple[set[str], bool]:
    sig = inspect.signature(fn)
    names = {n for n in sig.parameters if n != "self"}
    var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    return names, var_kw


@pytest.mark.parametrize("fake", [ApiFake, WorkerFake], ids=["record_api", "record_worker"])
def test_no_recorder_fake_invents_a_parameter_the_real_recorder_lacks(fake):
    invented: dict[str, set[str]] = {}
    for name, fn in inspect.getmembers(fake, predicate=inspect.isfunction):
        if name.startswith("_"):
            continue
        real = getattr(DatasetRecorder, name, None)
        if real is None or not inspect.isfunction(real):
            continue
        real_names, real_var_kw = _params(real)
        if real_var_kw:
            continue  # **kwargs accepts anything; a fake cannot lie about it
        fake_names, _ = _params(fn)
        extra = fake_names - real_names
        if extra:
            invented[name] = extra
    assert not invented, (
        f"{fake.__module__}.{fake.__qualname__} accepts parameters DatasetRecorder does not: "
        f"{invented}. A test using this fake would pass while the real recorder raised TypeError -- "
        "that is how Q56 (the Hub upload that never worked) survived."
    )


def test_the_audit_itself_can_fail():
    """A guard that cannot fail is decoration. This proves the check has teeth."""

    class LyingFake:
        def push_to_hub(self, repo_id=None):  # the exact Q56 shape
            ...

    with pytest.raises(AssertionError) as exc:
        test_no_recorder_fake_invents_a_parameter_the_real_recorder_lacks(LyingFake)
    assert "repo_id" in str(exc.value)


# ============================================================================
# from tests/test_dashboard_record_fps_declared.py
# Q54: the record form's fps must actually reach the session (backend half of the fix).
# ============================================================================


def _worker(**kw):
    class _Backend:
        """Only what the worker touches to report a session -- no arm, no frames."""

        cameras: dict = {}

        def leader_action(self):
            return {}

        def follower_apply(self, action):
            return {}

        def follower_observation(self):
            return {}

        def close(self):
            pass

    defaults = dict(
        dataset="cagatay/so101-pick",
        task="pick the cube",
        leader="so101-arm-2",
        follower="so101-arm-1",
        target_episodes=3,
        backend=_Backend(),
        recorder_factory=lambda **_: None,
        thumb_dir=pathlib.Path(tempfile.mkdtemp()),
        fps=30,
    )
    defaults.update(kw)
    return RecordWorker(**defaults)


@pytest.mark.parametrize("declared", [1, 4, 30, 60])
def test_the_declared_rate_is_what_the_session_reports(declared):
    w = _worker(fps=declared)
    assert w.fps == declared
    assert w.session()["fps"] == declared, "the panel reads this number back"


def test_the_worker_refuses_to_invent_a_rate():
    """The 30 lives in the ROUTE, deliberately: a worker with its own default would let a future
    caller forget the field and still produce a plausible-looking dataset."""
    import inspect

    sig = inspect.signature(RecordWorker.__init__)
    assert sig.parameters["fps"].default is inspect.Parameter.empty


def test_the_open_body_reading_matches_the_forms_contract():
    """The exact expression the route uses, pinned: `int(body.get("fps", 30) or 30)`.

    A 0 or an empty string means "no opinion" and must land on 30 rather than a rate that would
    divide by zero when timestamps are derived.
    """
    import inspect

    from strands_robots.dashboard import record_api

    src = inspect.getsource(record_api)
    assert 'fps=int(body.get("fps", 30) or 30)' in src


# ============================================================================
# from tests/test_dashboard_record_joint_gate.py
# The record gate refuses an arm that cannot say where it is (record_joints).
# ============================================================================

NOW = 1_000_000.0


def peer(joints, *, age=2.0):
    p = {"peer_id": "so101-leader", "last_seen": NOW - age}
    p["state"] = {"peer_id": "so101-leader", "t": NOW} if joints is None else {"joints": joints}
    return p


class TestItRefusesWhenThereIsRealEvidence:
    def test_a_fresh_snapshot_with_no_joints_is_refused_for_the_follower(self):
        r = record_joints.refusal(role="follower", peer_id="so101-follower", peer=peer(None), now=NOW)
        assert "so101-follower" in r and "NO joint positions" in r
        assert "observations" in r, "the follower's joints are the dataset's observations"
        assert "2s old" in r, "how old the evidence is, so the operator can judge it"

    def test_the_leader_is_gated_too_and_named_as_actions(self):
        r = record_joints.refusal(role="leader", peer_id="so101-leader", peer=peer({}), now=NOW)
        assert "actions" in r, "the leader's joints are the dataset's actions"

    def test_the_classified_reason_and_remedy_travel_with_it(self):
        r = record_joints.refusal(
            role="follower",
            peer_id="arm",
            peer=peer(None),
            now=NOW,
            problem={"headline": "this board has no calibration", "remedy": "Respawn it as leader_arm."},
        )
        assert "this board has no calibration." in r
        assert "Respawn it as leader_arm." in r
        assert "devices > logs" not in r, "the generic fallback must not tag along behind a real reason"

    def test_without_a_classified_reason_it_points_at_the_log(self):
        r = record_joints.refusal(role="follower", peer_id="arm", peer=peer(None), now=NOW)
        assert "devices > logs" in r


class TestItStaysQuietWithoutEvidence:
    """Each of these would block a legitimate recording, which is worse than the 500 it replaces."""

    def test_joints_present_proceeds(self):
        assert (
            record_joints.refusal(role="follower", peer_id="arm", peer=peer({"shoulder_pan.pos": 1.0}), now=NOW) is None
        )

    def test_no_snapshot_at_all_proceeds(self):
        for p in (None, {}, "not a mapping", 7):
            assert record_joints.refusal(role="follower", peer_id="arm", peer=p, now=NOW) is None

    def test_a_peer_with_no_state_block_proceeds(self):
        assert record_joints.refusal(role="follower", peer_id="arm", peer={"last_seen": NOW - 1}, now=NOW) is None

    def test_a_stale_snapshot_is_not_evidence_about_now(self):
        assert record_joints.refusal(role="follower", peer_id="arm", peer=peer(None, age=31.0), now=NOW) is None
        assert record_joints.refusal(role="follower", peer_id="arm", peer=peer(None, age=29.0), now=NOW) is not None

    def test_an_undateable_reading_is_not_evidence_either(self):
        p = peer(None)
        p.pop("last_seen")
        assert record_joints.refusal(role="follower", peer_id="arm", peer=p, now=NOW) is None
        p["last_seen"] = "yesterday"
        assert record_joints.refusal(role="follower", peer_id="arm", peer=p, now=NOW) is None

    def test_a_joints_shape_we_do_not_understand_is_not_absence(self):
        assert record_joints.refusal(role="follower", peer_id="arm", peer=peer("six"), now=NOW) is None


# ============================================================================
# from tests/test_dashboard_record_motion.py
# A recorded episode that never moves must SAY so (BUGS.md Q35).
# ============================================================================


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


# ============================================================================
# from tests/test_dashboard_record_still_life.py
# A still-life episode is reported BY THE WORKER, not just by the pure module (Q35 part 2).
# ============================================================================


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
        dataset="cagatay/so101-pick",
        task="pick up the cube",
        leader="arm-leader",
        follower="arm-follower",
        target_episodes=3,
        fps=5,
        backend=backend,
        recorder_factory=lambda **_: Recorder(),
        thumb_dir=str(tmp_path or "/tmp/rec-still-life-test"),
        clock=clock,
        autostart_loop=False,
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


# ============================================================================
# from tests/test_dashboard_record_target_name.py
# A taken dataset name is refused BEFORE the arms are touched (Q39).
# ============================================================================


def _home(monkeypatch, path) -> None:
    """Point dataset resolution at a temp home.

    NOT via the environment: ``_lerobot_home()`` prefers lerobot's own ``HF_LEROBOT_HOME``
    CONSTANT, which lerobot resolved when it was first imported - so setenv here changes nothing
    and the test would pass on a false negative (no dataset found => no refusal). Patching the
    function is the honest lever, and it is also the reason production must set that variable
    before the dashboard starts, not after.
    """
    from pathlib import Path

    from strands_robots import dataset_recorder

    monkeypatch.setattr(dataset_recorder, "_lerobot_home", lambda: Path(str(path)))


class TestTheVerdict:
    def test_no_name_at_all(self) -> None:
        for empty in ("", "   ", None):
            v = record_target_verdict(empty)  # type: ignore[arg-type]
            assert v and "dataset name is required" in v
            assert "repo_id" in v, "the refusal has to say what the name becomes"

    def test_a_free_name_is_not_refused(self) -> None:
        assert record_target_verdict("local/new-one", exists=False) is None

    def test_an_existing_dataset_with_episodes_names_what_is_at_stake(self) -> None:
        v = record_target_verdict("local/good", exists=True, has_meta=True, episodes=42)
        assert v is not None
        assert "42 recorded episode(s)" in v
        # BOTH halves of the truth: recording refuses, and overwriting destroys.
        assert "refuse" in v and "destroy" in v
        assert "Pick another name" in v

    def test_an_interrupted_session_s_leftovers_read_differently(self) -> None:
        # Same FileExistsError, different next action: there is nothing to lose here, so the
        # sentence must not imply the operator is about to destroy recorded work.
        v = record_target_verdict("local/half", exists=True, has_meta=True, episodes=0)
        assert v is not None
        assert "no recorded episodes" in v and "interrupted session" in v
        assert "destroy" not in v

    def test_a_non_dataset_directory_is_its_own_case(self) -> None:
        v = record_target_verdict("local/notes", exists=True, has_meta=False, non_empty=True)
        assert v is not None
        assert "not a dataset" in v
        assert "nothing here will delete files for you" in v

    def test_an_empty_directory_in_the_way_is_not_a_refusal(self) -> None:
        # An empty directory is what a resolve-then-mkdir dance leaves behind and LeRobot is happy
        # to write into it; refusing would invent a problem.
        assert record_target_verdict("local/x", exists=True, has_meta=False, non_empty=False) is None

    def test_an_unknown_episode_count_does_not_claim_a_number(self) -> None:
        v = record_target_verdict("local/x", exists=True, has_meta=True, episodes=None)
        assert v is not None and "no recorded episodes" in v


class TestTheFactsAreReadDefensively:
    def test_an_unreadable_home_yields_no_facts_rather_than_an_error(self, monkeypatch) -> None:
        from strands_robots.dashboard import record_api

        _home(monkeypatch, "/dev/null/nope")
        assert record_api._target_facts("local/whatever") in ({}, {"exists": False})

    def test_it_reports_a_real_dataset_on_disk(self, tmp_path, monkeypatch) -> None:
        from strands_robots.dashboard import record_api

        _home(monkeypatch, tmp_path)
        d = tmp_path / "local" / "taken" / "meta"
        d.mkdir(parents=True)
        (d / "info.json").write_text(json.dumps({"total_episodes": 7, "fps": 30}))
        facts = record_api._target_facts("local/taken")
        assert facts["exists"] is True and facts["has_meta"] is True and facts["episodes"] == 7
        # ...and the two halves compose into the sentence the operator sees.
        v = record_target_verdict("local/taken", **facts)
        assert v is not None and "7 recorded episode(s)" in v

    def test_an_empty_name_never_touches_the_disk(self) -> None:
        from strands_robots.dashboard import record_api

        assert record_api._target_facts("  ") == {}


def test_open_refuses_before_parking_any_arm(monkeypatch, tmp_path) -> None:
    """The point of the whole exercise: no despawn, no respawn, no 'could not open the arms'."""
    from fastapi import HTTPException

    from strands_robots.dashboard import record_api

    _home(monkeypatch, tmp_path)
    meta = tmp_path / "local" / "taken" / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({"total_episodes": 3, "fps": 30}))

    class Devices:
        def __init__(self) -> None:
            self.despawned: list[str] = []

        def despawn(self, peer_id: str) -> None:  # pragma: no cover - must not be reached
            self.despawned.append(peer_id)

    devices = Devices()
    ctl = record_api.RecordController(devices=devices, backend_factory=lambda **_: object())
    with pytest.raises(HTTPException) as err:
        ctl.open({"dataset": "local/taken", "task": "t", "leader": "a", "follower": "b"})

    assert err.value.status_code == 409
    assert "3 recorded episode(s)" in str(err.value.detail)
    assert "could not open the arms" not in str(err.value.detail)
    assert devices.despawned == [], "the arms were parked for a name collision"


# ============================================================================
# from tests/test_dashboard_record_upload_honesty.py
# Q56: the record panel's "upload to the Hugging Face Hub" tick had never published anything.
# ============================================================================


def test_a_successful_push_is_reported_from_the_recorders_own_answer():
    calls = []

    def push():
        calls.append(True)
        return {"status": "success", "repo_id": "cagatay/so101-pick", "episodes": 4}

    v = upload_verdict(asked_repo_id=None, dataset="cagatay/so101-pick", push=push)
    assert v == {"ok": True, "detail": "pushed to cagatay/so101-pick"}
    assert calls, "a plain upload must actually call the recorder"


def test_a_refused_push_is_never_reported_as_pushed():
    """The half that would have survived a naive repo_id fix."""

    def push():
        return {"status": "error", "message": "refusing to push empty dataset local/x (0 frames)"}

    v = upload_verdict(asked_repo_id=None, dataset="local/x", push=push)
    assert v["ok"] is False
    assert "REFUSED" in v["detail"]
    # The recorder's own reason travels: "upload failed" alone sends the operator nowhere.
    assert "empty dataset" in v["detail"]
    assert "pushed to" not in v["detail"]


def test_a_raising_push_says_the_dataset_is_still_on_disk():
    def push():
        raise RuntimeError("401 Client Error")

    v = upload_verdict(asked_repo_id=None, dataset="local/x", push=push)
    assert v["ok"] is False
    assert "saved locally" in v["detail"] and "401" in v["detail"]


def test_an_unreadable_answer_is_unknown_not_success():
    """Guessing "pushed" from a shape we do not recognise is the lie being removed."""
    v = upload_verdict(asked_repo_id=None, dataset="local/x", push=lambda: None)
    assert v["ok"] is False
    assert "UNKNOWN" in v["detail"]
    assert "pushed to" not in v["detail"]


def test_a_different_repo_id_is_refused_without_touching_the_hub():
    """A dataset publishes under the name it was recorded with; there is no argument for another.

    The old code passed the asked-for name as a kwarg that does not exist. Publishing under the
    recorded name instead would put the operator's episodes in a repo they did not name, so this
    refuses and says what would have to change.
    """
    calls = []

    v = upload_verdict(
        asked_repo_id="cagatay/other-name",
        dataset="cagatay/so101-pick",
        push=lambda: calls.append(True),
    )
    assert v["ok"] is False
    assert not calls, "the Hub must not be touched when the request cannot be honoured"
    assert "cagatay/so101-pick" in v["detail"] and "cagatay/other-name" in v["detail"]
    assert "saved" in v["detail"]


def test_the_same_repo_id_typed_out_is_not_a_conflict():
    """The UI defaults that box to the dataset name, so this is the common path."""
    v = upload_verdict(
        asked_repo_id="cagatay/so101-pick",
        dataset="cagatay/so101-pick",
        push=lambda: {"status": "success", "repo_id": "cagatay/so101-pick"},
    )
    assert v["ok"] is True


def test_the_recorder_is_still_called_with_no_repo_id_kwarg():
    """The literal defect: a kwarg the recorder's signature does not have.

    Read off the real signature so a future recorder that gains a repo_id argument makes this test
    fail loudly rather than leaving the dashboard on a stale assumption.
    """
    import inspect

    from strands_robots.dataset_recorder import DatasetRecorder

    params = inspect.signature(DatasetRecorder.push_to_hub).parameters
    assert "repo_id" not in params, (
        "push_to_hub gained a repo_id argument -- upload_verdict's refusal is now wrong and the "
        "record panel could honour a different name"
    )
