"""``get_recording_status`` reports the dataset, not the buffer a save just cleared.

Measured on ``Robot("so101", mode="sim")`` with a real LeRobot recording under
``root=``: ``stop_recording`` answered ``lab/ep -- 5 frames, 2 episode(s)`` and
the very next ``get_recording_status`` answered ``[idle] Not recording (last
episode: 0 steps)``. ``save_episode`` mid-session was the same lie one function
earlier - ``Episode 1 saved -- 3 frames`` followed by ``[recording] 0 steps
captured``.

Both writers reset ``state["trajectory"]``, and that buffer was the only thing
this reader counted, so every SUCCESSFUL save made it report nothing recorded -
the one moment a caller polls to confirm the save. What the dataset holds is now
stashed by :meth:`DatasetRecordingMixin._stash_saved_dataset` at both writers and
reported here, with the ``replay_episode`` call that reads it back.

The four facts are stashed as one self-consistent record rather than read from
the ``last_dataset_root`` / ``last_dataset_repo_id`` pair that ``start_recording``
writes: those move at the next ``start_recording`` while the counts describe the
previous save, so a hybrid read would name a dataset that was never saved (the
measured case is pinned below).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from strands_robots.simulation.recording import DatasetRecordingMixin


class _Engine(DatasetRecordingMixin):
    """Minimal concrete mixin host with a settable stub world (house shape)."""

    def __init__(self, world: Any = None) -> None:
        self._world = world


def _world(**backend_state: Any) -> SimpleNamespace:
    return SimpleNamespace(_backend_state=dict(backend_state))


class _Recorder:
    """A recorder shaped like the attributes the two writers read.

    ``save_episode`` and ``stop_recording`` are driven for real against this, so
    the cells below exercise the production write AND the production read rather
    than a state mapping the test set by hand - a suite that hand-sets
    ``last_dataset`` pins the reader and would survive deleting the write.
    """

    def __init__(self, repo_id: str = "lab/ep", root: str = "/tmp/x", *, pending: int = 3) -> None:
        self.repo_id = repo_id
        self.root = root
        self.episode_frame_count = pending
        self.frame_count = 0
        self.episode_count = 0
        self.dropped_frame_count = 0
        self.dataset = None
        self.finalized = False

    def save_episode(self) -> dict[str, Any]:
        self.frame_count += self.episode_frame_count
        self.episode_count += 1
        flushed, self.episode_frame_count = self.episode_frame_count, 0
        return {
            "status": "success",
            "episode": self.episode_count,
            "episode_frames": flushed,
            "total_frames": self.frame_count,
        }

    def finalize(self) -> None:
        self.finalized = True


def _recording_engine(recorder: _Recorder) -> _Engine:
    """An engine mid-recording, with the trajectory mirror the reader counts."""
    return _Engine(
        _world(
            recording=True,
            dataset_recorder=recorder,
            trajectory=[{"step": i} for i in range(recorder.episode_frame_count)],
            frames_at_start=0,
            episodes_at_start=0,
        )
    )


def _text(result: dict[str, Any]) -> str:
    return str(result["content"][0]["text"])


def _json(result: dict[str, Any]) -> dict[str, Any] | None:
    return next((c["json"] for c in result["content"] if "json" in c), None)


class TestASaveIsVisibleToTheNextPoll:
    def test_after_stop_recording_the_status_names_the_saved_dataset(self) -> None:
        """The measured defect: 5 frames saved, ``last episode: 0 steps`` reported."""
        engine = _recording_engine(_Recorder("lab/ep", "/tmp/x", pending=3))
        assert engine.stop_recording()["status"] == "success"

        text = _text(engine.get_recording_status())
        assert "lab/ep" in text
        assert "3 frames" in text and "1 episode(s)" in text
        assert "/tmp/x" in text
        assert "last episode: 0 steps" not in text

    def test_after_save_episode_the_open_episode_and_the_dataset_are_both_reported(self) -> None:
        """Mid-session: the buffer really is 0, the dataset really holds 3."""
        recorder = _Recorder("lab/ep", "/tmp/x", pending=3)
        engine = _recording_engine(recorder)
        assert engine.save_episode()["status"] == "success"

        text = _text(engine.get_recording_status())
        assert text.startswith("[recording] 0 steps captured")
        assert "3 frames" in text and "1 episode(s)" in text and "lab/ep" in text

    def test_the_status_names_the_call_that_reads_the_dataset_back(self) -> None:
        """The remedy is a call, spelled with the id that was saved.

        Executed literally against a real MuJoCo recording: the
        ``replay_episode(repo_id='lab/so101_ep')`` this text prints was parsed
        out of the message and run, and replayed 3/3 frames from the recorded
        root.
        """
        engine = _recording_engine(_Recorder("lab/so101_ep", "/tmp/x"))
        engine.stop_recording()
        assert "replay_episode(repo_id='lab/so101_ep')" in _text(engine.get_recording_status())

    def test_the_saved_dataset_is_machine_readable(self) -> None:
        engine = _recording_engine(_Recorder("lab/ep", "/tmp/x", pending=4))
        engine.stop_recording()

        payload = _json(engine.get_recording_status())
        assert payload is not None
        assert payload["recording"] is False
        assert payload["last_dataset"] == {
            "repo_id": "lab/ep",
            "root": "/tmp/x",
            "frame_count": 4,
            "episode_count": 1,
        }


class TestNothingSavedIsSaidAsNothingSaved:
    def test_a_session_that_saved_nothing_says_so(self) -> None:
        """``last episode: 0 steps`` was indistinguishable from a lost save."""
        text = _text(_Engine(_world(recording=False, trajectory=[])).get_recording_status())
        assert "nothing saved in this session" in text
        assert "last episode" not in text

    def test_an_open_recording_before_any_save_is_unchanged(self) -> None:
        """Control: with nothing saved yet there is no saved clause to add.

        Passes before and after the fix - the reply an in-progress poll already
        got must not grow a clause about a dataset that holds nothing.
        """
        engine = _recording_engine(_Recorder(pending=3))
        assert _text(engine.get_recording_status()) == "[recording] 3 steps captured"


class TestTheReportedFactsAgreeWithEachOther:
    def test_a_second_recording_does_not_relabel_what_was_saved(self) -> None:
        """The start-time seams move; the saved record must not follow them.

        ``_stash_dataset_target`` (``start_recording``) repoints
        ``last_dataset_repo_id`` / ``last_dataset_root`` at the NEW target. That
        pair is what a reader would naturally reach for, and the counts would
        still be the previous save's - naming a dataset with a frame count that
        was never written there. Measured on the real backend: the seams read
        ``lab/OTHER`` while the last saved dataset was ``lab/ep``.
        """
        engine = _recording_engine(_Recorder("lab/ep", "/tmp/x", pending=3))
        engine.stop_recording()
        engine._stash_dataset_target("lab/OTHER", "/tmp/other")

        assert engine._active_dataset_repo_id() == "lab/OTHER"
        text = _text(engine.get_recording_status())
        assert "lab/ep" in text and "/tmp/x" in text
        assert "lab/OTHER" not in text and "/tmp/other" not in text

    @pytest.mark.parametrize("root", [None, ""])
    def test_a_dataset_with_no_known_root_is_reported_without_one(self, root: str | None) -> None:
        """The id and counts still report; no ``at None`` in the text."""
        recorder = _Recorder("lab/ep", pending=2)
        recorder.root = root  # type: ignore[assignment]
        engine = _recording_engine(recorder)
        engine.stop_recording()

        text = _text(engine.get_recording_status())
        assert "lab/ep" in text and "2 frames" in text
        assert " at None" not in text and " at \n" not in text


class TestTheBookkeepingNeverBreaksTheSave:
    def test_a_recorder_that_cannot_name_its_dataset_still_saves(self) -> None:
        """The stash runs inside ``save_episode``; it must not be able to fail it.

        Minimal recorders (the fakes several backend suites drive ``save_episode``
        with, and any recorder predating the id) expose no ``repo_id``. Reading it
        as an attribute made the SAVE raise for the benefit of a later reader, so
        it is read defensively and an unnameable dataset is simply not reported.
        """
        recorder = _Recorder("lab/ep", "/tmp/x", pending=2)
        del recorder.repo_id
        engine = _recording_engine(recorder)

        assert engine.save_episode()["status"] == "success"
        assert engine._last_saved_dataset() is None
        assert _text(engine.get_recording_status()).startswith("[recording] 0 steps captured")


class TestTheOtherLifecycleStatesAreUntouched:
    def test_no_world_still_answers_the_create_world_remedy(self) -> None:
        assert "No world" in _text(_Engine(world=None).get_recording_status())
