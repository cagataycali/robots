"""``strands-robots verify-dataset`` - validate a recorded LeRobot dataset.

Detects the "mega-episode" corruption class: a collection run that intended to
record N distinct episodes but instead buffered every frame into a single
``episode_index=0`` episode (or whose ``meta/info.json`` count drifted from the
on-disk parquet). The parquet under ``meta/episodes/**/*.parquet`` is the ground
truth; this never trusts an agent's narration ("recorded 20/20") or in-memory
recorder bookkeeping.

Checks performed against a dataset root (the dir containing ``meta/``):
  1. parquet exists and holds at least one distinct episode;
  2. every episode has at least ``--min-frames`` frames (default 1) - flags any
     zero-length episode;
  3. ``meta/info.json`` ``total_episodes`` / ``total_frames`` (when present)
     agree with the parquet ground truth - flags metadata/parquet drift;
  4. when ``--expected N`` is given, the parquet holds exactly N episodes -
     flags the "wanted N, got M" mismatch.
  5. every per-episode video file referenced by the dataset (one MP4 per
     camera per episode, resolved from ``meta/info.json``'s ``video_path``
     template and the episode parquet's ``videos/<key>/chunk_index`` /
     ``file_index`` columns) exists on disk and is non-empty - flags the
     video-modality sibling of mega-episode corruption, where the episode
     count is correct but the pixels are missing/unwritten (disable with
     ``--no-check-videos``).

Usage:
    strands-robots verify-dataset /path/to/dataset
    strands-robots verify-dataset /path/to/dataset --expected 20
    python -m strands_robots verify-dataset ~/.cache/huggingface/lerobot/user/ds

Exit code is 0 when every check passes, 1 otherwise - so it drops straight into
CI as a dataset-integrity gate.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def verify_dataset(
    root: str | Path,
    expected: int | None = None,
    min_frames: int = 1,
    check_videos: bool = True,
) -> dict[str, Any]:
    """Verify the episode integrity of a LeRobot dataset on disk.

    Reads the canonical ``meta/episodes/**/*.parquet`` (via
    :func:`strands_robots.dataset_recorder.read_dataset_episode_indices`) and,
    when present, ``meta/info.json``, then runs the integrity checks described
    in the module docstring.

    Args:
        root: Dataset root directory (the dir that contains ``meta/``).
        expected: If given, require exactly this many distinct episodes.
        min_frames: Minimum frames every episode must contain (default 1).
        check_videos: When True (default), verify that every per-episode
            video file referenced by the dataset exists and is non-empty.

    Returns:
        A report dict with:
          - ``status``: ``"success"`` if every check passed, else ``"error"``.
          - ``ok``: bool mirror of ``status``.
          - ``root``: the resolved dataset root as a string.
          - ``total_episodes`` / ``total_frames`` / ``episode_indices`` /
            ``frames_per_episode``: parquet ground truth (zeros/empties when the
            dataset is empty or unreadable).
          - ``expected``: the requested count (or ``None``).
          - ``info_total_episodes`` / ``info_total_frames``: values declared in
            ``meta/info.json`` (``None`` when the file is absent or lacks them).
          - ``video_files_checked``: number of distinct per-episode video
            files resolved and checked (``0`` when ``check_videos`` is False
            or the dataset declares no video features).
          - ``problems``: list of human-readable failure strings (empty on pass).
    """
    from strands_robots.dataset_recorder import read_dataset_episode_indices

    root_path = Path(root)
    report: dict[str, Any] = {
        "status": "error",
        "ok": False,
        "root": str(root_path),
        "total_episodes": 0,
        "total_frames": 0,
        "episode_indices": [],
        "frames_per_episode": [],
        "expected": expected,
        "info_total_episodes": None,
        "info_total_frames": None,
        "video_files_checked": 0,
        "problems": [],
    }
    problems: list[str] = report["problems"]

    if expected is not None and (not isinstance(expected, int) or expected < 0):
        problems.append(f"expected must be a non-negative int, got {expected!r}")
        return report

    # Parquet ground truth.
    try:
        info = read_dataset_episode_indices(root_path)
    except FileNotFoundError as e:
        problems.append(str(e))
        return report
    except ImportError as e:
        problems.append(str(e))
        return report

    report["total_episodes"] = info["total_episodes"]
    report["total_frames"] = info["total_frames"]
    report["episode_indices"] = info["episode_indices"]
    report["frames_per_episode"] = info["frames_per_episode"]

    # Check 1: non-empty.
    if info["total_episodes"] == 0:
        problems.append("no episodes found in parquet (dataset is empty)")

    # Check 2: every episode has >= min_frames frames. Only when per-episode
    # lengths are available (the length column is optional in some writers).
    if min_frames > 0 and info["frames_per_episode"]:
        short = [
            (ep, n)
            for ep, n in zip(info["episode_indices"], info["frames_per_episode"], strict=False)
            if n < min_frames
        ]
        if short:
            detail = ", ".join(f"episode {ep}={n} frame(s)" for ep, n in short)
            problems.append(f"{len(short)} episode(s) below min_frames={min_frames}: {detail}")

    # Check 3: meta/info.json vs parquet ground truth (drift detection).
    info_json_path = root_path / "meta" / "info.json"
    if info_json_path.is_file():
        try:
            declared = json.loads(info_json_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            problems.append(f"could not read meta/info.json: {e}")
            declared = {}
        decl_eps = declared.get("total_episodes")
        decl_frames = declared.get("total_frames")
        report["info_total_episodes"] = decl_eps
        report["info_total_frames"] = decl_frames
        if isinstance(decl_eps, int) and decl_eps != info["total_episodes"]:
            problems.append(
                f"meta/info.json total_episodes={decl_eps} disagrees with parquet "
                f"({info['total_episodes']} distinct episode(s)) - metadata/parquet drift"
            )
        # Frame totals only meaningful when parquet carries per-episode lengths.
        if isinstance(decl_frames, int) and info["total_frames"] and decl_frames != info["total_frames"]:
            problems.append(
                f"meta/info.json total_frames={decl_frames} disagrees with parquet ({info['total_frames']} frame(s))"
            )

    # Check 4: exact expected episode count.
    if expected is not None and info["total_episodes"] != expected:
        problems.append(
            f"expected {expected} episode(s) but parquet holds {info['total_episodes']} "
            "- the recording did not produce the intended number of distinct episodes"
        )

    # Check 5: per-episode video files exist on disk and are non-empty. A
    # dataset can pass every count check yet carry missing/empty MP4 streams
    # (the recorder's video encoder failed, the files were partially synced, or
    # frames were never captured) - correct episode counts, no pixels. This is
    # the video-modality sibling of the mega-episode class above.
    if check_videos:
        checked, video_problems = _verify_video_files(root_path)
        report["video_files_checked"] = checked
        problems.extend(video_problems)

    report["ok"] = not problems
    report["status"] = "success" if report["ok"] else "error"
    return report


def _verify_video_files(root_path: Path) -> tuple[int, list[str]]:
    """Resolve and integrity-check every per-episode video file in a dataset.

    For each video feature declared in ``meta/info.json`` (``dtype == "video"``)
    and each episode, the on-disk MP4 path is resolved from the ``video_path``
    template and the episode parquet's ``videos/<key>/chunk_index`` /
    ``file_index`` columns, then checked for existence and non-zero size. This
    catches datasets whose episode counts are correct but whose pixels are
    missing or were never written.

    Args:
        root_path: Dataset root directory (the dir that contains ``meta/``).

    Returns:
        A ``(checked, problems)`` tuple where ``checked`` is the number of
        distinct video files resolved and ``problems`` lists missing/empty
        files (empty when the dataset declares no video features or all files
        are present and non-empty).
    """
    info_path = root_path / "meta" / "info.json"
    if not info_path.is_file():
        return 0, []
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # An unreadable info.json is already surfaced by the info.json drift
        # check; do not double-report it here.
        return 0, []

    features = info.get("features")
    if not isinstance(features, dict):
        return 0, []
    video_keys = [key for key, feat in features.items() if isinstance(feat, dict) and feat.get("dtype") == "video"]
    if not video_keys:
        return 0, []

    template = info.get("video_path")
    if not isinstance(template, str) or not template:
        return 0, [
            f"meta/info.json declares {len(video_keys)} video feature(s) but no 'video_path' "
            "template - cannot locate the per-episode MP4 files"
        ]

    try:
        import pyarrow.parquet as pq
    except ImportError:  # pragma: no cover - pyarrow ships with the lerobot extra
        return 0, []

    parquet_files = sorted((root_path / "meta" / "episodes").glob("**/*.parquet"))
    # (video_key, chunk_index, file_index) -> set of referencing episode_index.
    referenced: dict[tuple[str, int, int], int] = {}
    keys_with_refs: set[str] = set()
    for pf in parquet_files:
        table = pq.read_table(pf)
        cols = set(table.column_names)
        data = table.to_pydict()
        episodes = data.get("episode_index", [])
        for vk in video_keys:
            ci_col = f"videos/{vk}/chunk_index"
            fi_col = f"videos/{vk}/file_index"
            if ci_col not in cols or fi_col not in cols:
                continue
            keys_with_refs.add(vk)
            chunk_idx = data[ci_col]
            file_idx = data[fi_col]
            for i in range(len(chunk_idx)):
                ci, fi = chunk_idx[i], file_idx[i]
                if ci is None or fi is None:
                    continue
                ep = int(episodes[i]) if i < len(episodes) and episodes[i] is not None else -1
                referenced.setdefault((vk, int(ci), int(fi)), ep)

    problems: list[str] = []
    for vk in video_keys:
        if vk not in keys_with_refs:
            problems.append(
                f"video feature '{vk}' is declared but no episode references a video file for it "
                "(missing videos/<key>/chunk_index|file_index columns)"
            )

    for (vk, ci, fi), ep in sorted(referenced.items()):
        rel = template.format(video_key=vk, chunk_index=ci, file_index=fi)
        path = root_path / rel
        if not path.is_file():
            problems.append(f"missing video file for '{vk}' (episode {ep}): {rel}")
        elif path.stat().st_size == 0:
            problems.append(f"empty video file for '{vk}' (episode {ep}): {rel}")

    return len(referenced), problems


def _format_report(report: dict[str, Any]) -> str:
    """Render a report dict as a human-readable multi-line summary."""
    lines: list[str] = []
    verdict = "PASS" if report["ok"] else "FAIL"
    lines.append(f"[{verdict}] {report['root']}")
    lines.append(f"  episodes (parquet): {report['total_episodes']}")
    if report.get("video_files_checked"):
        lines.append(f"  video files checked: {report['video_files_checked']}")
    if report["total_frames"]:
        lines.append(f"  frames   (parquet): {report['total_frames']}")
    if report["expected"] is not None:
        lines.append(f"  expected episodes : {report['expected']}")
    if report["info_total_episodes"] is not None:
        lines.append(f"  info.json episodes: {report['info_total_episodes']}")
    for problem in report["problems"]:
        lines.append(f"  - {problem}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on success, 1 on any failed check."""
    parser = argparse.ArgumentParser(
        prog="strands-robots verify-dataset",
        description="Validate the episode integrity of a recorded LeRobot dataset.",
    )
    parser.add_argument("root", help="Dataset root directory (the dir that contains meta/).")
    parser.add_argument(
        "-e",
        "--expected",
        type=int,
        default=None,
        help="Require exactly this many distinct episodes (the count you intended to record).",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=1,
        help="Minimum frames every episode must contain (default: 1).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as JSON instead of the human-readable summary.",
    )
    parser.add_argument(
        "--no-check-videos",
        dest="check_videos",
        action="store_false",
        help="Skip per-episode video-file existence/non-empty checks.",
    )
    args = parser.parse_args(argv)

    report = verify_dataset(
        args.root, expected=args.expected, min_frames=args.min_frames, check_videos=args.check_videos
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(_format_report(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
