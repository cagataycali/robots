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


class TestMetadataEdgeCases:
    """Metadata-side validation paths beyond count/drift on a healthy dataset.

    These pin the verifier's behaviour on empties, frame-total drift, an
    unreadable ``meta/info.json``, a non-video feature set, and the missing
    optional dependency - the report's ``status`` / ``problems`` contract that
    drops the tool into CI as an integrity gate.
    """

    def test_empty_parquet_flagged_as_no_episodes(self, tmp_path: Path) -> None:
        # A finalized-but-empty dataset: the episodes parquet exists (so it is
        # not a FileNotFound) yet holds zero distinct episodes.
        _write_dataset(tmp_path, episode_indices=[])
        report = verify_dataset(tmp_path)
        assert report["status"] == "error"
        assert report["total_episodes"] == 0
        assert any("dataset is empty" in p for p in report["problems"])

    def test_info_json_frame_total_drift_flagged(self, tmp_path: Path) -> None:
        # Episode count agrees but the declared frame total does not - isolates
        # the total_frames drift check from the total_episodes one.
        _write_dataset(
            tmp_path,
            episode_indices=[0, 1],
            frames_per_episode=[3, 3],
            info={"total_episodes": 2, "total_frames": 99},
        )
        report = verify_dataset(tmp_path)
        assert report["status"] == "error"
        assert any("total_frames=99" in p and "6 frame(s)" in p for p in report["problems"])

    def test_unreadable_info_json_reported_once(self, tmp_path: Path) -> None:
        # A corrupt info.json is surfaced by the drift check; the video check
        # silently degrades on the same unreadable file rather than double-
        # reporting it.
        _write_dataset(tmp_path, episode_indices=[0, 1], frames_per_episode=[3, 3])
        (tmp_path / "meta" / "info.json").write_text("{not valid json", encoding="utf-8")
        report = verify_dataset(tmp_path)
        assert report["status"] == "error"
        assert sum("could not read meta/info.json" in p for p in report["problems"]) == 1
        assert report["video_files_checked"] == 0

    def test_non_video_features_check_nothing(self, tmp_path: Path) -> None:
        # info.json declares features but none are videos (state-only dataset):
        # the video check resolves zero files without complaint.
        _write_dataset(
            tmp_path,
            episode_indices=[0, 1],
            frames_per_episode=[4, 4],
            info={
                "total_episodes": 2,
                "total_frames": 8,
                "features": {"observation.state": {"dtype": "float32", "shape": [6]}},
            },
        )
        report = verify_dataset(tmp_path)
        assert report["status"] == "success", report["problems"]
        assert report["video_files_checked"] == 0

    def test_missing_pyarrow_reported_as_problem(self, tmp_path: Path, monkeypatch) -> None:
        # The lerobot extra (pyarrow) is absent: the ground-truth read raises
        # ImportError, which the verifier turns into a problem rather than a
        # traceback.
        import strands_robots.dataset_recorder as dr

        def _raise(_root):
            raise ImportError("read_dataset_episode_indices requires pyarrow (installed with the lerobot extra).")

        monkeypatch.setattr(dr, "read_dataset_episode_indices", _raise)
        report = verify_dataset(tmp_path)
        assert report["status"] == "error"
        assert any("pyarrow" in p for p in report["problems"])


class TestVideoIndexColumnEdgeCases:
    """Video-reference resolution when the parquet index columns are absent or null."""

    def test_video_feature_without_index_columns_flagged(self, tmp_path: Path) -> None:
        # The feature is declared but the parquet carries no
        # videos/<key>/chunk_index|file_index columns, so no episode references
        # a file for it - the verifier flags the dangling declaration.
        _write_video_dataset(
            tmp_path,
            episode_indices=[0, 1],
            video_keys=["observation.images.cam1"],
            frames_per_episode=[3, 3],
            include_index_columns=False,
        )
        report = verify_dataset(tmp_path)
        assert report["status"] == "error"
        assert report["video_files_checked"] == 0
        assert any("no episode references a video file" in p for p in report["problems"])

    def test_null_video_index_rows_skipped(self, tmp_path: Path) -> None:
        # A row whose chunk_index/file_index is null references no file and is
        # skipped, not resolved into a phantom missing-file problem.
        vk = "observation.images.cam1"
        ep_dir = tmp_path / "meta" / "episodes" / "chunk-000"
        ep_dir.mkdir(parents=True)
        pq.write_table(
            pa.table(
                {
                    "episode_index": [0],
                    "length": [3],
                    f"videos/{vk}/chunk_index": pa.array([None], type=pa.int64()),
                    f"videos/{vk}/file_index": pa.array([None], type=pa.int64()),
                }
            ),
            ep_dir / "episodes_000.parquet",
        )
        (tmp_path / "meta" / "info.json").write_text(
            json.dumps(
                {
                    "total_episodes": 1,
                    "total_frames": 3,
                    "features": {vk: {"dtype": "video", "shape": [3, 64, 64]}},
                    "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                }
            ),
            encoding="utf-8",
        )
        report = verify_dataset(tmp_path)
        # No file resolved (the only row is null) and no missing-file problem.
        assert report["video_files_checked"] == 0
        assert not any("video file" in p for p in report["problems"])


def _write_dataset_with_stats(
    root: Path,
    episode_indices: list[int],
    counts: list[int],
    action_minmax: list[tuple[list[float], list[float]]] | None = None,
    state_minmax: list[tuple[list[float], list[float]]] | None = None,
    info: dict | None = None,
) -> Path:
    """Write an episodes parquet carrying the inline per-episode feature stats.

    LeRobot v3 stores ``stats/<feature>/min`` / ``.../max`` / ``.../count`` as
    list-typed columns (one row per episode). This builds exactly that schema so
    the dead-control-column check can be exercised without lerobot or mujoco.

    Args:
        root: Dataset root (the dir that will contain ``meta/``).
        episode_indices: Distinct ``episode_index`` values to write.
        counts: Per-episode frame count (the ``count`` stat).
        action_minmax: Per-episode ``(min_vec, max_vec)`` for ``action`` (omit
            the column when None).
        state_minmax: Per-episode ``(min_vec, max_vec)`` for
            ``observation.state`` (omit the column when None).
        info: Optional ``meta/info.json`` payload.

    Returns:
        The dataset root path.
    """
    ep_dir = root / "meta" / "episodes" / "chunk-000"
    ep_dir.mkdir(parents=True, exist_ok=True)
    columns: dict[str, list] = {
        "episode_index": episode_indices,
        "length": counts,
    }
    if action_minmax is not None:
        columns["stats/action/min"] = [mn for mn, _ in action_minmax]
        columns["stats/action/max"] = [mx for _, mx in action_minmax]
        columns["stats/action/count"] = [[c] for c in counts]
    if state_minmax is not None:
        columns["stats/observation.state/min"] = [mn for mn, _ in state_minmax]
        columns["stats/observation.state/max"] = [mx for _, mx in state_minmax]
        columns["stats/observation.state/count"] = [[c] for c in counts]
    pq.write_table(pa.table(columns), ep_dir / "episodes_000.parquet")
    if info is not None:
        (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    return root


class TestDeadControlColumn:
    """All-zero ``action`` / ``observation.state`` columns are flagged.

    A recording can pass every count, length and video check yet carry a control
    column written entirely as zeros (a writer whose action keys never resolved
    to the declared columns). The per-episode stats expose this cheaply.
    """

    def test_all_zero_action_column_flagged(self, tmp_path: Path) -> None:
        _write_dataset_with_stats(
            tmp_path,
            episode_indices=[0],
            counts=[10],
            action_minmax=[([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])],
        )
        report = verify_dataset(tmp_path, check_videos=False)
        assert report["ok"] is False
        assert report["stats_vectors_checked"] == 1
        assert any("identically zero" in p and "action" in p for p in report["problems"])

    def test_varying_action_column_passes(self, tmp_path: Path) -> None:
        _write_dataset_with_stats(
            tmp_path,
            episode_indices=[0, 1],
            counts=[10, 12],
            action_minmax=[
                ([-0.5, -0.1, 0.0], [0.5, 0.2, 0.3]),
                ([-0.4, 0.0, -0.2], [0.4, 0.1, 0.2]),
            ],
        )
        report = verify_dataset(tmp_path, check_videos=False)
        assert report["ok"] is True
        assert report["stats_vectors_checked"] == 2
        assert report["problems"] == []

    def test_all_zero_observation_state_flagged(self, tmp_path: Path) -> None:
        _write_dataset_with_stats(
            tmp_path,
            episode_indices=[0],
            counts=[8],
            state_minmax=[([0.0, 0.0], [0.0, 0.0])],
        )
        report = verify_dataset(tmp_path, check_videos=False)
        assert report["ok"] is False
        assert any("observation.state" in p and "identically zero" in p for p in report["problems"])

    def test_single_frame_all_zero_not_flagged(self, tmp_path: Path) -> None:
        # A single-frame episode has min == max trivially; all-zero there is not
        # yet evidence of a dead column, so it must not be flagged.
        _write_dataset_with_stats(
            tmp_path,
            episode_indices=[0],
            counts=[1],
            action_minmax=[([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])],
        )
        report = verify_dataset(tmp_path, check_videos=False)
        assert report["ok"] is True
        assert report["problems"] == []

    def test_no_check_stats_skips_dead_column(self, tmp_path: Path) -> None:
        _write_dataset_with_stats(
            tmp_path,
            episode_indices=[0],
            counts=[10],
            action_minmax=[([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])],
        )
        report = verify_dataset(tmp_path, check_videos=False, check_stats=False)
        assert report["ok"] is True
        assert report["stats_vectors_checked"] == 0

    def test_dataset_without_stats_columns_passes(self, tmp_path: Path) -> None:
        # Writers that omit inline stats must not be penalized.
        _write_dataset(tmp_path, episode_indices=[0, 1], frames_per_episode=[5, 5])
        report = verify_dataset(tmp_path, check_videos=False)
        assert report["ok"] is True
        assert report["stats_vectors_checked"] == 0

    def test_one_dead_episode_among_healthy_flagged(self, tmp_path: Path) -> None:
        _write_dataset_with_stats(
            tmp_path,
            episode_indices=[0, 1, 2],
            counts=[10, 10, 10],
            action_minmax=[
                ([-0.5, -0.1], [0.5, 0.2]),
                ([0.0, 0.0], [0.0, 0.0]),  # dead
                ([-0.3, -0.2], [0.3, 0.2]),
            ],
        )
        report = verify_dataset(tmp_path, check_videos=False)
        assert report["ok"] is False
        dead = [p for p in report["problems"] if "identically zero" in p]
        assert len(dead) == 1
        assert "episode 1" in dead[0]

    def test_dead_action_cli_exit_code(self, tmp_path: Path) -> None:
        _write_dataset_with_stats(
            tmp_path,
            episode_indices=[0],
            counts=[10],
            action_minmax=[([0.0, 0.0], [0.0, 0.0])],
        )
        assert verify_main([str(tmp_path), "--no-check-videos"]) == 1
        assert verify_main([str(tmp_path), "--no-check-videos", "--no-check-stats"]) == 0
