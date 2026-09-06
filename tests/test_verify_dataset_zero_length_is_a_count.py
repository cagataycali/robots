"""A recorded length of zero is a frame count, not a missing ``length`` column.

:func:`~strands_robots.dataset_recorder.read_dataset_episode_indices` reports
per-episode frame counts, and an empty ``frames_per_episode`` is documented to
mean the ``length`` column is unavailable. It scored that availability as
``any(f > 0 for f in frames_per_episode)``, so the one dataset whose every
episode recorded zero frames answered "this dataset carries no lengths" - and
:func:`~strands_robots.verify_dataset.verify_dataset` reads an empty list as
"nothing to compare" and skips its check 2, the check whose entire subject is a
zero-length episode (module docstring, item 2).

The gate was therefore non-monotonic in the damage it grades. A dataset holding
``[5, 0, 0]`` frames named its two empty episodes and exited 1; the strictly
worse ``[0, 0, 0]`` - a whole collection run that wrote metadata and no frames -
was certified ``status="success"``, CLI exit 0, and the header claiming
``total_frames: 900`` beside it went unreported too, because the drift half of
check 3 was gated on the parquet total being non-zero rather than on a length
being available. Adding an episode with real frames was the way to make the gate
notice.

Availability is now whether a length was *read*: a present column with a
recorded ``0`` is a count, while an absent column - and one present but wholly
null, a length that is unknown rather than zero - stays unavailable, so neither
gains a false zero-length verdict. ``min_frames=0`` remains the documented way
to skip the length check.

Fixtures are pyarrow-only (no lerobot, no mujoco) and every assertion is on
observable behaviour: the report's ``status`` / ``problems``, the CLI exit code,
and the counts the reader is handed.
"""

from __future__ import annotations

import re
import types
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pyarrow")

from strands_robots.dataset_recorder import read_dataset_episode_indices
from strands_robots.simulation.base import SimEngine
from strands_robots.verify_dataset import main as verify_main
from strands_robots.verify_dataset import verify_dataset

from .test_verify_dataset import _write_dataset

#: A header claiming frames the parquet does not hold, so check 3's frame-drift
#: half has something to disagree with in every dataset below.
_DECLARES_900_FRAMES = {"total_episodes": 3, "total_frames": 900}


def _verify(root: Path, **kwargs: Any) -> dict[str, Any]:
    """Run the gate on the count checks alone (no video or stats fixtures)."""
    return verify_dataset(root, check_videos=False, check_stats=False, **kwargs)


def _named_short_episodes(report: dict[str, Any]) -> str:
    """The check-2 problem string, or ``""`` when the check did not report one."""
    return next((p for p in report["problems"] if "frame(s)" in p and "min_frames" in p), "")


def _short_episode_indices(report: dict[str, Any]) -> list[int]:
    """Which episodes check 2 actually named as short, in the order reported."""
    return [int(n) for n in re.findall(r"episode (\d+)=", _named_short_episodes(report))]


def _frame_drift(report: dict[str, Any]) -> str:
    """The check-3 frame-total problem string, or ``""`` when absent."""
    return next((p for p in report["problems"] if "total_frames=900 disagrees" in p), "")


class TestEveryEpisodeZeroLength:
    """The whole-run corruption is graded, not read as an absent column."""

    def test_the_recorded_zeros_are_reported_as_counts(self, tmp_path: Path) -> None:
        info = read_dataset_episode_indices(_write_dataset(tmp_path, [0, 1, 2], [0, 0, 0]))
        assert info["frames_per_episode"] == [0, 0, 0]
        assert info["total_frames"] == 0
        assert info["total_episodes"] == 3

    def test_the_gate_names_every_empty_episode(self, tmp_path: Path) -> None:
        report = _verify(_write_dataset(tmp_path, [0, 1, 2], [0, 0, 0]), min_frames=1)
        assert report["status"] == "error", report
        problem = _named_short_episodes(report)
        assert "3 episode(s) below min_frames=1" in problem, report["problems"]
        for episode in (0, 1, 2):
            assert f"episode {episode}=0 frame(s)" in problem

    def test_the_header_that_claims_frames_is_reported(self, tmp_path: Path) -> None:
        """Check 3's frame comparison survives a parquet total of zero."""
        root = _write_dataset(tmp_path, [0, 1, 2], [0, 0, 0], info=_DECLARES_900_FRAMES)
        report = _verify(root, min_frames=1)
        assert "parquet (0 frame(s))" in _frame_drift(report), report["problems"]

    def test_the_cli_exits_non_zero(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        _write_dataset(tmp_path, [0, 1, 2], [0, 0, 0])
        assert verify_main([str(tmp_path), "--no-check-videos", "--no-check-stats"]) == 1
        assert "min_frames" in capsys.readouterr().out

    def test_the_worse_dataset_is_not_the_one_that_passes(self, tmp_path: Path) -> None:
        """Monotonicity: emptying the one good episode cannot buy a pass."""
        mixed = _verify(_write_dataset(tmp_path / "mixed", [0, 1, 2], [5, 0, 0]), min_frames=1)
        emptied = _verify(_write_dataset(tmp_path / "emptied", [0, 1, 2], [0, 0, 0]), min_frames=1)
        assert mixed["status"] == "error"
        assert emptied["status"] == "error"
        assert _short_episode_indices(mixed) == [1, 2]
        assert _short_episode_indices(emptied) == [0, 1, 2]

    def test_a_single_zero_length_episode_is_graded_too(self, tmp_path: Path) -> None:
        """The degenerate one-episode run, where every length is also zero."""
        report = _verify(_write_dataset(tmp_path, [0], [0]), min_frames=1)
        assert report["status"] == "error", report
        assert "episode 0=0 frame(s)" in _named_short_episodes(report)


class TestAnUnavailableLengthStaysUnavailable:
    """A length nobody recorded is not a zero, and must not be graded as one."""

    def test_an_absent_column_reports_no_lengths(self, tmp_path: Path) -> None:
        info = read_dataset_episode_indices(_write_dataset(tmp_path, [0, 1, 2]))
        assert info["frames_per_episode"] == []
        assert info["total_frames"] == 0

    def test_a_wholly_null_column_reports_no_lengths(self, tmp_path: Path) -> None:
        """Present but never written: unknown per episode, not zero."""
        info = read_dataset_episode_indices(_write_dataset(tmp_path, [0, 1, 2], [None, None, None]))
        assert info["frames_per_episode"] == []
        assert info["total_frames"] == 0

    @pytest.mark.parametrize(
        "lengths",
        [pytest.param(None, id="absent-column"), pytest.param([None, None, None], id="wholly-null-column")],
    )
    def test_no_length_check_verdict_is_invented(self, tmp_path: Path, lengths: list[Any] | None) -> None:
        report = _verify(_write_dataset(tmp_path, [0, 1, 2], lengths), min_frames=1)
        assert report["status"] == "success", report
        assert _named_short_episodes(report) == ""

    @pytest.mark.parametrize(
        "lengths",
        [pytest.param(None, id="absent-column"), pytest.param([None, None, None], id="wholly-null-column")],
    )
    def test_no_frame_drift_is_invented(self, tmp_path: Path, lengths: list[Any] | None) -> None:
        """Nothing to compare the header against, so nothing is reported."""
        root = _write_dataset(tmp_path, [0, 1, 2], lengths, info=_DECLARES_900_FRAMES)
        report = _verify(root, min_frames=1)
        assert _frame_drift(report) == "", report["problems"]

    def test_one_recorded_length_makes_the_column_available(self, tmp_path: Path) -> None:
        """A partially-written column is read, and its nulls read as zero."""
        info = read_dataset_episode_indices(_write_dataset(tmp_path, [0, 1], [None, 4]))
        assert info["frames_per_episode"] == [0, 4]
        assert info["total_frames"] == 4


class TestTheDocumentedSkipAndTheHealthyCaseAreUntouched:
    """The two behaviours a reader of this gate already relies on."""

    def test_min_frames_zero_still_skips_the_length_check(self, tmp_path: Path) -> None:
        report = _verify(_write_dataset(tmp_path, [0, 1, 2], [0, 0, 0]), min_frames=0)
        assert report["status"] == "success", report
        assert _named_short_episodes(report) == ""

    def test_a_healthy_dataset_still_passes(self, tmp_path: Path) -> None:
        root = _write_dataset(tmp_path, [0, 1, 2], [5, 5, 5], info={"total_episodes": 3, "total_frames": 15})
        assert _verify(root, min_frames=1)["status"] == "success"

    def test_an_agreeing_frame_total_is_still_not_a_problem(self, tmp_path: Path) -> None:
        root = _write_dataset(tmp_path, [0, 1, 2], [5, 5, 5], info={"total_episodes": 3, "total_frames": 15})
        assert _verify(root, min_frames=1)["problems"] == []


class TestTheEpisodeCountFacadeReportsTheZerosItDoesNotGrade:
    """``verify_dataset_episodes`` answers the count question, and only that.

    Its verdict is deliberately left on the episode count and the two metadata
    sources agreeing - the frame lengths are diagnostics there, and the gate
    above owns the length verdict. What changes is that the diagnostic is now
    populated for this dataset: an operator reading ``total_frames_per_ep``
    sees three empty episodes rather than the empty list that used to mean the
    recording carried no lengths at all.
    """

    @staticmethod
    def _facade_json(root: Path, expected: int) -> dict[str, Any]:
        stub = types.SimpleNamespace(_active_dataset_root=lambda: root)
        result = SimEngine.verify_dataset_episodes(stub, expected)  # type: ignore[arg-type]
        return next(block["json"] for block in result["content"] if "json" in block)

    def test_the_per_episode_diagnostic_shows_the_empty_episodes(self, tmp_path: Path) -> None:
        payload = self._facade_json(_write_dataset(tmp_path, [0, 1, 2], [0, 0, 0]), 3)
        assert payload["total_frames_per_ep"] == [0, 0, 0]
        assert payload["total_frames"] == 0

    def test_the_count_verdict_is_unchanged(self, tmp_path: Path) -> None:
        """The boundary, stated: three episodes recorded is three episodes."""
        root = _write_dataset(tmp_path, [0, 1, 2], [0, 0, 0], info={"total_episodes": 3})
        stub = types.SimpleNamespace(_active_dataset_root=lambda: root)
        result = SimEngine.verify_dataset_episodes(stub, 3)  # type: ignore[arg-type]
        assert result["status"] == "success"
