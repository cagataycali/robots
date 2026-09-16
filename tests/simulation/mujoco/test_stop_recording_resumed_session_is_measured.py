"""A resumed recording session is measured against the dataset it resumed.

``DatasetRecorder.resume`` seeds ``frame_count`` / ``episode_count`` with the
dataset's totals so the report reflects the whole dataset. On its own that
counter cannot say whether THIS session captured anything: a resumed session
that captured nothing passed the empty-capture guard on the previous sessions'
frames and stop_recording answered "Episode saved … 19 frames, 1 episode(s)"
for an episode that was never written. Measured on Robot("so101", mode="sim"):
record one episode, start_recording again into the same root, stop at once.

start_recording now stashes the counts it started from; stop_recording refuses
a session that appended nothing (naming the unchanged dataset) and, when it
did append, says how much of the total is this session's. A fresh dataset is
reported exactly as before.
"""

from __future__ import annotations

import pytest

from strands_robots.simulation import Simulation

from .test_stop_recording_finalize import _arm, _FakeRecorder


@pytest.fixture
def recording_sim():
    s = Simulation(tool_name="resumed_session_test", mesh=False)
    s.create_world()
    yield s
    s.cleanup()


def _resumed(sim, recorder, *, frames_at_start: int, episodes_at_start: int) -> None:
    _arm(sim, recorder)
    sim._world._backend_state["frames_at_start"] = frames_at_start
    sim._world._backend_state["episodes_at_start"] = episodes_at_start


class TestAResumedSessionThatCapturedNothing:
    def test_is_refused_and_names_the_unchanged_dataset(self, recording_sim) -> None:
        rec = _FakeRecorder(frame_count=19, episode_frame_count=0)  # 19 = the dataset's own frames
        _resumed(recording_sim, rec, frames_at_start=19, episodes_at_start=1)
        result = recording_sim.stop_recording()
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "This session captured no frames" in text
        assert "local/finalize_test (19 frames, 1 episode(s)) is unchanged" in text
        assert "no episode was saved" in text
        # the recipe the fresh-dataset refusal gives is still there, verbatim
        assert "captured no frames - dataset would be empty" in text and "run_policy" in text
        assert rec.calls == [], "nothing to flush, nothing to finalize"
        assert recording_sim._world._backend_state["dataset_recorder"] is None

    def test_the_stash_does_not_outlive_the_refused_session(self, recording_sim) -> None:
        """The refusal releases the recorder, so the counts describing it go too."""
        rec = _FakeRecorder(frame_count=19, episode_frame_count=0)
        _resumed(recording_sim, rec, frames_at_start=19, episodes_at_start=1)
        assert recording_sim.stop_recording()["status"] == "error"
        state = recording_sim._world._backend_state
        assert "frames_at_start" not in state and "episodes_at_start" not in state

    def test_the_stash_is_cleared_when_the_recorder_is_released(self, recording_sim) -> None:
        rec = _FakeRecorder(frame_count=25, episode_frame_count=6)
        _resumed(recording_sim, rec, frames_at_start=19, episodes_at_start=1)
        assert recording_sim.stop_recording()["status"] == "success"
        state = recording_sim._world._backend_state
        assert "frames_at_start" not in state and "episodes_at_start" not in state


class TestAResumedSessionThatAppended:
    def test_the_report_separates_this_session_from_the_total(self, recording_sim) -> None:
        rec = _FakeRecorder(frame_count=25, episode_frame_count=6)
        rec.episode_count = 2  # the dataset's 1 + the episode this session flushes
        _resumed(recording_sim, rec, frames_at_start=19, episodes_at_start=1)
        result = recording_sim.stop_recording()
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "25 frames, 2 episode(s) (+6 frames, +1 episode(s) this session)" in text


class TestAFreshDatasetReadsAsBefore:
    def test_no_session_note_without_a_resume(self, recording_sim) -> None:
        rec = _FakeRecorder(frame_count=7, episode_frame_count=7)
        _arm(recording_sim, rec)
        text = recording_sim.stop_recording()["content"][0]["text"]
        assert "this session" not in text
        assert "7 frames" in text

    def test_an_empty_fresh_dataset_keeps_the_plain_refusal(self, recording_sim) -> None:
        rec = _FakeRecorder(frame_count=0, episode_frame_count=0)
        _arm(recording_sim, rec)
        text = recording_sim.stop_recording()["content"][0]["text"]
        assert text.startswith("stop_recording captured no frames")
        assert "This session" not in text


class TestOnTheRealRecorder:
    """The whole path on a real LeRobotDataset in tmp_path: record, resume-empty, resume-append."""

    @staticmethod
    def _text(result) -> str:
        return " ".join(c.get("text", "") for c in result["content"] if isinstance(c, dict))

    def test_record_then_empty_resume_then_append(self, tmp_path) -> None:
        pytest.importorskip("lerobot")
        from strands_robots import Robot

        sim = Robot("so101", mode="sim")

        def start() -> dict:
            """One recording target, opened three times: fresh, resumed, resumed."""
            return sim.start_recording(root=str(tmp_path), task="t", fps=30, repo_id="lab/resumed")

        try:
            assert start()["status"] == "success"
            sim.set_joint_positions(robot_name="so101", positions=[0.3, 0, 0, 0, 0, 0], hold=True)
            sim.step(n_steps=300)
            first = self._text(sim.stop_recording())
            assert "1 episode(s)" in first and "this session" not in first

            resumed = self._text(start())
            assert "Resuming the existing dataset (1 episode(s)," in resumed
            assert "overwrite=True" in resumed
            empty = sim.stop_recording()
            assert empty["status"] == "error"
            assert "This session captured no frames" in self._text(empty)
            assert "is unchanged and no episode was saved" in self._text(empty)

            assert start()["status"] == "success"
            sim.step(n_steps=300)
            appended = sim.stop_recording()
            assert appended["status"] == "success"
            assert "2 episode(s) (+" in self._text(appended) and "+1 episode(s) this session" in self._text(appended)
        finally:
            sim.cleanup()
