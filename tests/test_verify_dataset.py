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


def _write_video_dataset(
    root: Path,
    episode_indices: list[int],
    video_keys: list[str],
    *,
    frames_per_episode: list[int] | None = None,
    write_files: set[str] | None = None,
    empty_files: set[str] | None = None,
    video_path: str | None = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    include_index_columns: bool = True,
) -> Path:
    """Write a synthetic LeRobot v3 dataset that declares video features.

    Each episode references one MP4 per camera (chunk 0, file = episode index)
    via the ``videos/<key>/chunk_index`` / ``file_index`` columns, mirroring the
    real recorder layout. The actual MP4 files are written only for the
    ``(video_key, episode)`` pairs in ``write_files`` (default: all), so a test
    can simulate a missing or empty video stream.

    Args:
        root: Dataset root (the dir that will contain ``meta/``).
        episode_indices: Distinct ``episode_index`` values to write.
        video_keys: Video feature keys (e.g. ``observation.images.cam``).
        frames_per_episode: Per-episode ``length`` column (omitted if None).
        write_files: ``{"<video_key>:<episode>"}`` pairs whose MP4 is written
            (default: every pair).
        empty_files: Subset of ``write_files`` written as 0-byte files.
        video_path: ``meta/info.json`` ``video_path`` template (None to omit).
        include_index_columns: Whether to emit the per-key chunk/file columns.

    Returns:
        The dataset root path.
    """
    ep_dir = root / "meta" / "episodes" / "chunk-000"
    ep_dir.mkdir(parents=True, exist_ok=True)
    columns: dict[str, list] = {"episode_index": episode_indices}
    if frames_per_episode is not None:
        columns["length"] = frames_per_episode
    if include_index_columns:
        for vk in video_keys:
            columns[f"videos/{vk}/chunk_index"] = [0 for _ in episode_indices]
            columns[f"videos/{vk}/file_index"] = list(episode_indices)
    pq.write_table(pa.table(columns), ep_dir / "episodes_000.parquet")

    features = {
        vk: {"dtype": "video", "shape": [3, 64, 64], "names": ["channels", "height", "width"]} for vk in video_keys
    }
    info: dict = {
        "total_episodes": len(episode_indices),
        "total_frames": sum(frames_per_episode) if frames_per_episode else 0,
        "features": features,
    }
    if video_path is not None:
        info["video_path"] = video_path
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")

    if write_files is None:
        write_files = {f"{vk}:{ep}" for vk in video_keys for ep in episode_indices}
    empty_files = empty_files or set()
    for pair in write_files:
        vk, ep = pair.rsplit(":", 1)
        rel = video_path.format(video_key=vk, chunk_index=0, file_index=int(ep)) if video_path else None
        if rel is None:
            continue
        fpath = root / rel
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_bytes(b"" if pair in empty_files else b"\x00\x00\x00\x18ftypmp42")
    return root


class TestVideoFileIntegrity:
    """Check 5: per-episode video files must exist on disk and be non-empty.

    A dataset can pass every episode-count check yet carry missing or empty MP4
    streams - correct episode counts, no pixels. These pin that the verifier
    catches that video-modality corruption, and that ``check_videos=False``
    opts out (the pre-Check-5 behaviour).
    """

    def test_all_videos_present_passes(self, tmp_path: Path) -> None:
        _write_video_dataset(
            tmp_path,
            episode_indices=[0, 1],
            video_keys=["observation.images.cam1", "observation.images.cam2"],
            frames_per_episode=[3, 3],
        )
        report = verify_dataset(tmp_path)
        assert report["status"] == "success", report["problems"]
        # 2 cameras x 2 episodes = 4 distinct video files resolved and checked.
        assert report["video_files_checked"] == 4

    def test_missing_video_file_fails(self, tmp_path: Path) -> None:
        # Episode 1's cam2 MP4 is never written; episode counts are still valid.
        keys = ["observation.images.cam1", "observation.images.cam2"]
        all_pairs = {f"{k}:{e}" for k in keys for e in (0, 1)}
        _write_video_dataset(
            tmp_path,
            episode_indices=[0, 1],
            video_keys=keys,
            frames_per_episode=[3, 3],
            write_files=all_pairs - {"observation.images.cam2:1"},
        )
        report = verify_dataset(tmp_path)
        assert report["status"] == "error"
        assert report["total_episodes"] == 2  # count check still passes
        assert any("missing video file" in p and "cam2" in p for p in report["problems"])

    def test_empty_video_file_fails(self, tmp_path: Path) -> None:
        _write_video_dataset(
            tmp_path,
            episode_indices=[0],
            video_keys=["observation.images.cam1"],
            frames_per_episode=[3],
            empty_files={"observation.images.cam1:0"},
        )
        report = verify_dataset(tmp_path)
        assert report["status"] == "error"
        assert any("empty video file" in p for p in report["problems"])

    def test_check_videos_false_skips_the_check(self, tmp_path: Path) -> None:
        # The exact pre-Check-5 behaviour: a missing MP4 is invisible when the
        # video check is disabled, so the count-only verdict is success.
        keys = ["observation.images.cam1"]
        _write_video_dataset(
            tmp_path,
            episode_indices=[0],
            video_keys=keys,
            frames_per_episode=[3],
            write_files=set(),  # write no MP4 files at all
        )
        report = verify_dataset(tmp_path, check_videos=False)
        assert report["status"] == "success"
        assert report["video_files_checked"] == 0

    def test_no_video_features_checks_nothing(self, tmp_path: Path) -> None:
        # A state-only dataset (no video features) has nothing to check.
        _write_dataset(tmp_path, episode_indices=[0, 1], frames_per_episode=[4, 4])
        report = verify_dataset(tmp_path)
        assert report["status"] == "success"
        assert report["video_files_checked"] == 0

    def test_video_feature_without_path_template_fails(self, tmp_path: Path) -> None:
        _write_video_dataset(
            tmp_path,
            episode_indices=[0],
            video_keys=["observation.images.cam1"],
            frames_per_episode=[3],
            video_path=None,
        )
        report = verify_dataset(tmp_path)
        assert report["status"] == "error"
        assert any("no 'video_path' template" in p for p in report["problems"])

    def test_cli_no_check_videos_flag_skips(self, tmp_path: Path, capsys) -> None:
        _write_video_dataset(
            tmp_path,
            episode_indices=[0],
            video_keys=["observation.images.cam1"],
            frames_per_episode=[3],
            write_files=set(),
        )
        assert verify_main([str(tmp_path)]) == 1  # missing MP4 fails by default
        capsys.readouterr()
        assert verify_main([str(tmp_path), "--no-check-videos"]) == 0  # opt out passes
