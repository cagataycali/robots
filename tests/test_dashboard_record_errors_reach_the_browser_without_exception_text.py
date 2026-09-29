"""A recorder failure reaches the operator's screen as a kind or a reason, never as exception text.

``DatasetRecorder`` answers a failed ``save_episode`` or ``push_to_hub`` with the
exception's own words in ``message``: a path, a token in a Hub URL, a traceback
line. The recorder is a library and its callers (the run_policy tool, a script)
want those words, so ``message`` keeps them. The dashboard is the one caller
whose reader is a browser, so ``record_worker`` derives the browser's sentence
from the two other keys the recorder now sets, ``reason`` (the recorder's own
short sentence for a refusal it decided itself) and ``error_type`` (the class of
the exception a call raised), and logs ``message`` in full.

``contained_path`` is also pinned here in the shape a path checker reads as a
barrier: a prefix test alone, then the separator test alone. The pair admits
exactly what the earlier compound test did, and the sibling directory case is
the one that would tell them apart.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from unittest import mock

import pytest
from fastapi import HTTPException

from strands_robots.dashboard import routes_record
from strands_robots.dashboard.record_worker import recorder_error_summary, upload_verdict

SECRET = "hf_secret_token_in_url /srv/datasets/private/path Traceback (most recent call last)"


class TestTheRecorderNamesItsErrorsTwice:
    def test_a_refusal_carries_its_own_reason(self) -> None:
        from strands_robots.dataset_recorder import DatasetRecorder

        recorder = DatasetRecorder.__new__(DatasetRecorder)
        recorder.frame_count = 0
        recorder.episode_count = 0
        recorder.dataset = mock.Mock(repo_id="org/empty")
        out = recorder.push_to_hub()
        assert out["status"] == "error"
        assert out["reason"] == "empty dataset (0 frames, 0 episodes)"
        assert "refusing to push empty dataset org/empty" in out["message"]

    def test_a_hub_failure_carries_the_exceptions_class_beside_its_words(self) -> None:
        from strands_robots.dataset_recorder import DatasetRecorder

        recorder = DatasetRecorder.__new__(DatasetRecorder)
        recorder.frame_count = 10
        recorder.episode_count = 1
        recorder.dataset = mock.Mock(repo_id="org/ds")
        recorder.dataset.push_to_hub.side_effect = PermissionError(SECRET)
        out = recorder.push_to_hub()
        assert out == {"status": "error", "message": SECRET, "error_type": "PermissionError"}

    def test_a_failed_save_carries_the_exceptions_class_beside_its_words(self) -> None:
        from strands_robots.dataset_recorder import DatasetRecorder

        recorder = DatasetRecorder.__new__(DatasetRecorder)
        recorder._closed = False
        recorder.dataset = mock.Mock()
        recorder.dataset.save_episode.side_effect = OSError(SECRET)
        out = recorder.save_episode()
        assert out["status"] == "error"
        assert out["error_type"] == "OSError"
        assert SECRET in out["message"]  # the library keeps the words for its other callers
        assert recorder._closed is True


class TestTheBrowserSentenceIsBuiltFromNeitherMessageNorException:
    @pytest.mark.parametrize(
        ("info", "expected"),
        [
            (
                {"status": "error", "message": SECRET, "reason": "empty dataset (0 frames, 0 episodes)"},
                "empty dataset (0 frames, 0 episodes)",
            ),
            ({"status": "error", "message": SECRET, "error_type": "PermissionError"}, "PermissionError"),
            ({"status": "error", "message": SECRET}, "no reason given"),
            ({"status": "error", "message": SECRET, "reason": "  ", "error_type": "OSError"}, "OSError"),
            ({"status": "error", "message": SECRET, "reason": 7, "error_type": None}, "no reason given"),
        ],
    )
    def test_the_summary_prefers_reason_then_kind_and_never_quotes_message(self, info: dict, expected: str) -> None:
        summary = recorder_error_summary(info)
        assert summary == expected
        assert "hf_secret" not in summary and "Traceback" not in summary

    def test_upload_verdict_logs_the_words_and_shows_the_kind(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.ERROR, logger="strands_robots.dashboard.record_worker"):
            verdict = upload_verdict(
                asked_repo_id=None,
                dataset="org/ds",
                push=lambda: {"status": "error", "message": SECRET, "error_type": "HfHubHTTPError"},
            )
        assert verdict == {"ok": False, "detail": "saved locally, upload REFUSED: HfHubHTTPError"}
        assert any("hf_secret_token_in_url" in rec.getMessage() for rec in caplog.records)

    def test_upload_verdict_shows_the_recorders_own_refusal(self) -> None:
        verdict = upload_verdict(
            asked_repo_id=None,
            dataset="org/ds",
            push=lambda: {"status": "error", "message": SECRET, "reason": "empty dataset (0 frames, 0 episodes)"},
        )
        assert verdict["detail"] == "saved locally, upload REFUSED: empty dataset (0 frames, 0 episodes)"

    def test_a_failed_save_reaches_the_session_as_its_kind(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from strands_robots.dashboard import record_worker

        worker = record_worker.RecordWorker.__new__(record_worker.RecordWorker)
        # the smallest worker stop_episode() reads: one episode in flight with frames
        worker._lock = __import__("threading").RLock()
        worker._closed = False
        worker._phase = "recording"
        worker._current = mock.Mock(frames=3, started_at=0.0)
        worker._motion = []
        worker._motion_notice = None
        monkeypatch.setattr(record_worker.record_motion, "motion_verdict", lambda *_a, **_k: None)
        worker._episodes = []
        worker._clock = lambda: 1.0
        worker._last_error = None
        worker._recorder = mock.Mock()
        worker._recorder.save_episode.return_value = {"status": "error", "message": SECRET, "error_type": "OSError"}
        worker.session = lambda: {"error": worker._last_error}  # type: ignore[method-assign]
        with caplog.at_level(logging.ERROR, logger="strands_robots.dashboard.record_worker"):
            out = worker.stop_episode()
        assert out == {"error": "save_episode failed: OSError"}
        assert any("hf_secret_token_in_url" in rec.getMessage() for rec in caplog.records)


class TestContainedPathAdmitsTheSameSetInTheBarrierShape:
    @pytest.fixture()
    def homes(self, tmp_path: Path) -> tuple[Path, Path, Path]:
        home = tmp_path / "home"
        (home / "ds").mkdir(parents=True)
        sibling = tmp_path / "home2"
        (sibling / "x").mkdir(parents=True)
        outside = tmp_path / "out"
        outside.mkdir()
        (home / "link").symlink_to(outside)
        return Path(os.path.realpath(home)), sibling, outside

    def test_the_home_and_its_children_are_admitted(self, homes: tuple[Path, Path, Path]) -> None:
        home, _, _ = homes
        with mock.patch.object(routes_record, "dataset_home", lambda: home):
            assert routes_record.contained_path(str(home)) == home
            assert routes_record.contained_path(str(home / "ds")) == home / "ds"
            assert routes_record.contained_path(str(home / "ds" / ".." / "ds")) == home / "ds"
            assert routes_record.contained_path(str(home / "ds") + os.sep) == home / "ds"

    @pytest.mark.parametrize("which", ["sibling", "sibling_child", "symlink_out", "dotdot_out"])
    def test_everything_else_is_refused_without_naming_the_target(
        self, homes: tuple[Path, Path, Path], which: str
    ) -> None:
        home, sibling, outside = homes
        raw = {
            "sibling": str(sibling),
            "sibling_child": str(sibling / "x"),
            "symlink_out": str(home / "link"),
            "dotdot_out": str(home / ".." / "out"),
        }[which]
        with mock.patch.object(routes_record, "dataset_home", lambda: home), pytest.raises(HTTPException) as e:
            routes_record.contained_path(raw)
        assert e.value.status_code == 400
        assert str(outside) not in str(e.value.detail) and str(sibling) not in str(e.value.detail)

    @pytest.mark.parametrize("raw", ["", "   ", None, 3])
    def test_a_missing_path_is_a_422_not_a_containment_refusal(self, raw: object) -> None:
        with pytest.raises(HTTPException) as e:
            routes_record.contained_path(raw)
        assert e.value.status_code == 422
