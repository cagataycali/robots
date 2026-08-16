"""The dataset-integrity gate refuses a threshold it cannot honor.

:func:`strands_robots.verify_dataset.verify_dataset` and the sim-facade
:meth:`~strands_robots.simulation.base.SimEngine.verify_dataset_episodes` ask
one question - does this recording hold the episodes it claims - and take two
discrete counts to ask it. ``expected`` carried its own comparison on both
surfaces and ``min_frames`` carried none, so the threshold that decides check 2
was the one input nothing checked.

The consequence is specific to a gate: ``min_frames`` is compared as
``min_frames > 0`` before the check runs, so a value that fails that comparison
does not fail loudly - it *switches the check off*. A dataset holding a
zero-length episode, the exact corruption class the check exists to detect,
was certified ``status="success"`` (CLI exit 0) under ``min_frames=-5``,
``False`` or ``nan``, while ``"2"`` / ``None`` / ``[2]`` escaped as a bare
``TypeError`` past a checker documented to always produce a report.

Both counts now share :func:`~strands_robots.utils.non_negative_count_error`,
which keeps ``0`` first-class - ``min_frames=0`` is the documented way to skip
the length check, and ``expected=0`` asks that a dataset be empty - and rejects
``bool``, closing a hole both hand-rolled copies shared: as an ``int`` subclass
``True`` passed ``isinstance(value, int) and value >= 0`` and became a silent
threshold, or a silent episode count, of one.

Fixtures are pyarrow-only (no lerobot, no mujoco) and every assertion is on
observable behaviour: the report's ``status`` / ``problems``, whether the
zero-length episode was actually named, and the CLI exit code.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pyarrow")

from strands_robots.simulation.base import SimEngine
from strands_robots.utils import non_negative_count_error
from strands_robots.verify_dataset import main as verify_main
from strands_robots.verify_dataset import verify_dataset

from .test_verify_dataset import _write_dataset

# Values no episode-count threshold can be honored as. ``-5`` is refused by the
# sibling ``expected`` on both surfaces, so it also measures the divergence
# between two counts in one signature.
UNUSABLE_COUNTS: list[Any] = [-5, -1, True, False, 2.7, 3.0, float("nan"), float("inf"), "2", [2], {"n": 2}]

# Usable: a real threshold, and the documented "skip this check" spelling.
USABLE_COUNTS: list[Any] = [0, 1, 5]


def _corrupt(root: Path) -> Path:
    """A dataset whose episode 1 holds zero frames - check 2's whole purpose."""
    return _write_dataset(root, episode_indices=[0, 1], frames_per_episode=[5, 0])


def _healthy(root: Path) -> Path:
    """A dataset every check passes."""
    return _write_dataset(
        root,
        episode_indices=[0, 1],
        frames_per_episode=[5, 5],
        info={"total_episodes": 2, "total_frames": 10},
    )


def _verify(**kwargs: Any) -> dict[str, Any]:
    """Call the gate with deliberately off-type values without narrowing them.

    ``min_frames`` / ``expected`` are annotated ``int``; the point of these
    tests is what the runtime does with a value an agent or a shell can still
    supply, so the arguments are splatted rather than passed positionally.
    """
    return verify_dataset(**kwargs)


def _flagged_short_episode(report: dict[str, Any]) -> bool:
    """Did the report actually name the zero-length episode?

    Distinguished from "a problem mentioning ``min_frames`` exists", because a
    refusal of the threshold also names it. Only check 2 reports a frame count.
    """
    return any("frame(s)" in p for p in report["problems"])


class TestAThresholdOutsideTheDomainIsRefusedNotApplied:
    """``min_frames`` is reported, and never silently disables the check."""

    @pytest.mark.parametrize("value", UNUSABLE_COUNTS)
    def test_the_report_names_the_threshold_as_the_problem(self, tmp_path: Path, value: Any) -> None:
        report = _verify(root=_corrupt(tmp_path), min_frames=value)
        assert report["status"] == "error", report
        assert report["ok"] is False
        assert any("min_frames must be a non-negative integer" in p for p in report["problems"]), report["problems"]

    @pytest.mark.parametrize("value", UNUSABLE_COUNTS)
    def test_a_corrupt_dataset_is_never_certified(self, tmp_path: Path, value: Any) -> None:
        """The headline: no threshold may turn this gate into a pass."""
        report = _verify(root=_corrupt(tmp_path), min_frames=value)
        assert report["status"] != "success", report

    @pytest.mark.parametrize("value", UNUSABLE_COUNTS)
    def test_nothing_escapes_as_a_traceback(self, tmp_path: Path, value: Any) -> None:
        """The checker's documented contract is that a bad input yields a report."""
        report = _verify(root=_corrupt(tmp_path), min_frames=value)
        assert isinstance(report, dict) and isinstance(report["problems"], list)

    def test_the_threshold_is_reported_before_the_parquet_is_read(self, tmp_path: Path) -> None:
        """A refused threshold is a caller error, not a dataset verdict."""
        report = _verify(root=_corrupt(tmp_path), min_frames=-5)
        assert report["total_episodes"] == 0
        assert report["episode_indices"] == []
        assert not _flagged_short_episode(report)


class TestZeroStaysTheDocumentedSkipSpelling:
    """``min_frames=0`` skips check 2; it is the only spelling that does."""

    def test_zero_still_disables_the_length_check(self, tmp_path: Path) -> None:
        report = _verify(root=_corrupt(tmp_path), min_frames=0)
        assert report["status"] == "success", report
        assert not _flagged_short_episode(report)

    @pytest.mark.parametrize("value", [1, 5])
    def test_a_real_threshold_still_flags_the_short_episode(self, tmp_path: Path, value: int) -> None:
        report = _verify(root=_corrupt(tmp_path), min_frames=value)
        assert report["status"] == "error"
        assert _flagged_short_episode(report), report["problems"]

    @pytest.mark.parametrize("value", USABLE_COUNTS)
    def test_a_healthy_dataset_still_passes(self, tmp_path: Path, value: int) -> None:
        report = _verify(root=_healthy(tmp_path), min_frames=value)
        assert report["status"] == "success", report


class TestTheSiblingCountSharesTheDomain:
    """``expected`` keeps its behaviour and loses its ``bool`` hole."""

    def test_a_boolean_episode_count_is_refused(self, tmp_path: Path) -> None:
        """``expected=True`` used to certify a one-episode dataset."""
        report = _verify(root=_write_dataset(tmp_path, [0], [3]), expected=True)
        assert report["status"] == "error"
        assert any("expected must be a non-negative integer" in p for p in report["problems"])

    @pytest.mark.parametrize("value", UNUSABLE_COUNTS)
    def test_every_unusable_expected_is_reported(self, tmp_path: Path, value: Any) -> None:
        report = _verify(root=_write_dataset(tmp_path, [0], [3]), expected=value)
        assert any("expected must be a non-negative integer" in p for p in report["problems"]), report["problems"]

    def test_none_still_means_no_expected_check(self, tmp_path: Path) -> None:
        report = _verify(root=_healthy(tmp_path), expected=None)
        assert report["status"] == "success", report

    def test_zero_still_asks_for_an_empty_dataset(self, tmp_path: Path) -> None:
        report = _verify(root=_write_dataset(tmp_path, [0], [3]), expected=0)
        assert report["status"] == "error"
        assert any("expected" in p and "0" in p for p in report["problems"])

    def test_a_matching_count_still_passes(self, tmp_path: Path) -> None:
        report = _verify(root=_healthy(tmp_path), expected=2)
        assert report["status"] == "success", report

    def test_the_existing_message_wording_is_preserved(self, tmp_path: Path) -> None:
        """The pre-existing pin reads ``"non-negative int"``, a substring here."""
        report = _verify(root=_write_dataset(tmp_path, [0], [3]), expected=-1)
        assert any("non-negative int" in p for p in report["problems"])


def _episodes_verdict(value: Any) -> str:
    """Classify the sim facade's answer without needing a recorded dataset."""
    stub = types.SimpleNamespace(_active_dataset_root=lambda: None)
    result = SimEngine.verify_dataset_episodes(stub, value)  # type: ignore[arg-type]
    text = "".join(block.get("text", "") for block in result["content"])
    return "refused" if "must be a non-negative integer" in text else "accepted"


class TestBothSurfacesAgreeOnTheDomain:
    """One question, one accepted domain, whichever surface asks it."""

    @pytest.mark.parametrize("value", UNUSABLE_COUNTS)
    def test_neither_surface_accepts_an_unusable_count(self, tmp_path: Path, value: Any) -> None:
        gate = _verify(root=_write_dataset(tmp_path, [0], [3]), expected=value)
        gate_refused = any("expected must be a non-negative integer" in p for p in gate["problems"])
        assert gate_refused, gate["problems"]
        assert _episodes_verdict(value) == "refused"

    @pytest.mark.parametrize("value", USABLE_COUNTS)
    def test_neither_surface_refuses_a_usable_count(self, tmp_path: Path, value: int) -> None:
        gate = _verify(root=_healthy(tmp_path), expected=value)
        assert not any("must be a non-negative integer" in p for p in gate["problems"]), gate["problems"]
        assert _episodes_verdict(value) == "accepted"

    def test_the_facade_message_still_names_the_method(self, tmp_path: Path) -> None:
        stub = types.SimpleNamespace(_active_dataset_root=lambda: None)
        result = SimEngine.verify_dataset_episodes(stub, -1)  # type: ignore[arg-type]
        assert "verify_dataset_episodes: expected must be a non-negative int" in result["content"][0]["text"]

    def test_the_shared_helper_owns_the_rule(self) -> None:
        """Non-vacuity: the domain under test is the shared helper's, not a copy."""
        for value in UNUSABLE_COUNTS:
            assert non_negative_count_error(value, "expected", "verify_dataset") is not None, value
        for value in USABLE_COUNTS:
            assert non_negative_count_error(value, "expected", "verify_dataset") is None, value


class TestTheCliGateFailsOnARefusedThreshold:
    """A shell-invoked CI gate must not exit 0 on a refused argument."""

    def test_a_negative_min_frames_exits_nonzero(self, tmp_path: Path, capsys: Any) -> None:
        """``--min-frames -5`` used to exit 0 on a dataset with a 0-frame episode."""
        code = verify_main([str(_corrupt(tmp_path)), "--min-frames", "-5"])
        assert code == 1
        assert "min_frames must be a non-negative integer" in capsys.readouterr().out

    def test_zero_min_frames_still_exits_zero(self, tmp_path: Path, capsys: Any) -> None:
        code = verify_main([str(_corrupt(tmp_path)), "--min-frames", "0"])
        capsys.readouterr()
        assert code == 0

    def test_the_default_still_catches_the_short_episode(self, tmp_path: Path, capsys: Any) -> None:
        code = verify_main([str(_corrupt(tmp_path))])
        assert code == 1
        assert "frame(s)" in capsys.readouterr().out
