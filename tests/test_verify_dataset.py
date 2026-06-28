"""Episode-integrity verification for recorded LeRobot datasets.

Pins the contract of ``strands_robots.verify_dataset`` - the dataset-integrity
gate that detects the "mega-episode" corruption class (a run that intended N
episodes but buffered everything into one ``episode_index=0`` episode) and
``meta/info.json`` vs parquet drift.

Fixtures are built with pyarrow directly so the tests need neither lerobot nor
mujoco: they write the canonical ``meta/episodes/**/*.parquet`` (and optional
``meta/info.json``) that :func:`read_dataset_episode_indices` reads as ground
truth. Each test asserts observable behaviour - the report's ``status`` /
``problems`` and the CLI exit code - not internal state.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

import pyarrow as pa
import pyarrow.parquet as pq

from strands_robots.verify_dataset import main as verify_main
from strands_robots.verify_dataset import verify_dataset


def _write_dataset(
    root: Path,
    episode_indices: list[int],
    frames_per_episode: list[int] | None = None,
    info: dict | None = None,
) -> Path:
    """Write a minimal LeRobot v3 dataset (episodes parquet + optional info.json).

    Args:
        root: Dataset root (the dir that will contain ``meta/``).
        episode_indices: Distinct ``episode_index`` values to write.
        frames_per_episode: Per-episode ``length`` column (omitted if None).
        info: Optional ``meta/info.json`` payload.

    Returns:
        The dataset root path.
    """
    ep_dir = root / "meta" / "episodes" / "chunk-000"
    ep_dir.mkdir(parents=True, exist_ok=True)
    columns: dict[str, list] = {"episode_index": episode_indices}
    if frames_per_episode is not None:
        columns["length"] = frames_per_episode
    pq.write_table(pa.table(columns), ep_dir / "episodes_000.parquet")
    if info is not None:
        (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    return root


class TestVerifyDatasetHealthy:
    """A well-formed multi-episode dataset passes every check."""

    def test_three_episodes_with_matching_info_passes(self, tmp_path: Path) -> None:
        _write_dataset(
            tmp_path,
            episode_indices=[0, 1, 2],
            frames_per_episode=[3, 3, 3],
            info={"total_episodes": 3, "total_frames": 9},
        )
        report = verify_dataset(tmp_path)
        assert report["status"] == "success"
        assert report["ok"] is True
        assert report["problems"] == []
        assert report["total_episodes"] == 3
        assert report["total_frames"] == 9

    def test_expected_count_match_passes(self, tmp_path: Path) -> None:
        _write_dataset(tmp_path, episode_indices=[0, 1, 2], frames_per_episode=[5, 5, 5])
        report = verify_dataset(tmp_path, expected=3)
        assert report["status"] == "success"


class TestMegaEpisodeCorruption:
    """The headline bug: one merged episode where N were intended."""

    def test_single_episode_against_expected_20_fails(self, tmp_path: Path) -> None:
        _write_dataset(tmp_path, episode_indices=[0], frames_per_episode=[60])
        report = verify_dataset(tmp_path, expected=20)
        assert report["status"] == "error"
        assert report["total_episodes"] == 1
        assert any("expected 20" in p and "holds 1" in p for p in report["problems"])

    def test_info_json_claims_20_but_parquet_has_1_fails(self, tmp_path: Path) -> None:
        # No --expected supplied: the metadata/parquet drift alone is the signal.
        _write_dataset(
            tmp_path,
            episode_indices=[0],
            frames_per_episode=[60],
            info={"total_episodes": 20, "total_frames": 60},
        )
        report = verify_dataset(tmp_path)
        assert report["status"] == "error"
        assert any("info.json total_episodes=20" in p for p in report["problems"])


class TestEdgeCases:
    """Empty datasets, zero-length episodes, and bad arguments."""

    def test_missing_dataset_reports_filenotfound(self, tmp_path: Path) -> None:
        report = verify_dataset(tmp_path / "does_not_exist")
        assert report["status"] == "error"
        assert report["total_episodes"] == 0
        assert report["problems"]

    def test_zero_length_episode_flagged(self, tmp_path: Path) -> None:
        _write_dataset(tmp_path, episode_indices=[0, 1], frames_per_episode=[5, 0])
        report = verify_dataset(tmp_path, min_frames=1)
        assert report["status"] == "error"
        assert any("min_frames" in p for p in report["problems"])

    def test_negative_expected_rejected(self, tmp_path: Path) -> None:
        _write_dataset(tmp_path, episode_indices=[0], frames_per_episode=[3])
        report = verify_dataset(tmp_path, expected=-1)
        assert report["status"] == "error"
        assert any("non-negative int" in p for p in report["problems"])


class TestCLI:
    """The ``verify-dataset`` CLI maps pass/fail to exit codes 0/1."""

    def test_cli_exit_zero_on_healthy(self, tmp_path: Path, capsys) -> None:
        _write_dataset(tmp_path, episode_indices=[0, 1], frames_per_episode=[4, 4])
        rc = verify_main([str(tmp_path)])
        assert rc == 0
        assert "PASS" in capsys.readouterr().out

    def test_cli_exit_one_on_mismatch(self, tmp_path: Path, capsys) -> None:
        _write_dataset(tmp_path, episode_indices=[0], frames_per_episode=[60])
        rc = verify_main([str(tmp_path), "--expected", "20"])
        assert rc == 1
        assert "FAIL" in capsys.readouterr().out

    def test_cli_json_output_parses(self, tmp_path: Path, capsys) -> None:
        _write_dataset(tmp_path, episode_indices=[0, 1, 2], frames_per_episode=[3, 3, 3])
        rc = verify_main([str(tmp_path), "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["total_episodes"] == 3
        assert payload["ok"] is True
