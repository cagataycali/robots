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

    report["ok"] = not problems
    report["status"] = "success" if report["ok"] else "error"
    return report


def _format_report(report: dict[str, Any]) -> str:
    """Render a report dict as a human-readable multi-line summary."""
    lines: list[str] = []
    verdict = "PASS" if report["ok"] else "FAIL"
    lines.append(f"[{verdict}] {report['root']}")
    lines.append(f"  episodes (parquet): {report['total_episodes']}")
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
    args = parser.parse_args(argv)

    report = verify_dataset(args.root, expected=args.expected, min_frames=args.min_frames)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(_format_report(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
