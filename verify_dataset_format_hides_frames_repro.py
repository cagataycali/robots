"""Minimal repro + fix verification for verify-dataset _format_report asymmetry.

The verifier is marketed as "parquet is the ground truth; the checker never
trusts an agent's narration" (docs/learn/data/verify.md).  The CLI's human
report hid TWO frame facts the user needs:

  1) ``frames (parquet)`` was suppressed when total_frames == 0, which is
     the signature of the "mega-episode / save_episode never flushed real
     frames" failure mode the verifier exists to detect.

  2) ``info.json frames`` was NEVER rendered, even on PASS or when the
     header disagreed. The sibling line ``info.json episodes`` IS always
     rendered. The pair was asymmetric; a user who saw "frames (parquet):
     270" had to read the ``problems`` list to learn what info.json
     claimed.

Upstream site:  strands_robots/verify_dataset.py:704-723
Docs sample:    docs/learn/data/verify.md:34-40 (sample report)

Run:  python verify_dataset_format_hides_frames_repro.py
Pass criterion (after fix): the three expected lines are present in the
emitted text.  Set env ``PROVE_PREFIX_BUG=1`` to flip the assertions and
reproduce the pre-fix behavior.
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq


def _mk_dataset(root: pathlib.Path, episodes: list[tuple[int, int]],
                info_total_episodes: int | None,
                info_total_frames: int | None) -> None:
    if root.exists():
        shutil.rmtree(root)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({
            "episode_index": [e for e, _ in episodes],
            "length":        [n for _, n in episodes],
        }),
        root / "meta/episodes/chunk-000/file-000.parquet",
    )
    info: dict = {"fps": 30, "features": {}}
    if info_total_episodes is not None:
        info["total_episodes"] = info_total_episodes
    if info_total_frames is not None:
        info["total_frames"] = info_total_frames
    (root / "meta/info.json").write_text(json.dumps(info))


def _run_cli(root: pathlib.Path, *extra: str) -> str:
    out = subprocess.run(
        ["strands-robots", "verify-dataset", str(root), *extra],
        check=False, capture_output=True, text=True, timeout=60,
    )
    return out.stdout


PRE_FIX = os.environ.get("PROVE_PREFIX_BUG") == "1"


def case_1_zero_frames_mega_episode() -> None:
    """A dataset of one episode-of-length-0 is EXACTLY the failure the
    verifier exists to catch.  Pre-fix the CLI printed episodes but NOT frames.
    """
    root = pathlib.Path("/tmp/verify_fmt_case1")
    _mk_dataset(root, episodes=[(0, 0)],
                info_total_episodes=1, info_total_frames=0)
    text = _run_cli(root, "--expected", "1")
    print("=== Case 1: zero-frame episode (mega-episode signature) ===")
    print(text)
    assert "episodes (parquet): 1" in text
    if PRE_FIX:
        assert "frames   (parquet):" not in text, "pre-fix: frames line hidden"
        print(">>> PRE-FIX CONFIRMED: 'frames (parquet)' line is suppressed.\n")
    else:
        assert "frames   (parquet): 0" in text, \
            "post-fix: frames (parquet): 0 must appear"
        print(">>> FIX OK: 'frames (parquet): 0' is now rendered.\n")


def case_2_info_frames_rendering() -> None:
    """info.json total_episodes was shown; info.json total_frames was NEVER."""
    root = pathlib.Path("/tmp/verify_fmt_case2")
    _mk_dataset(root, episodes=[(0, 90), (1, 90), (2, 90)],
                info_total_episodes=5, info_total_frames=450)
    text = _run_cli(root)
    print("=== Case 2: info.json headers disagree with parquet ===")
    print(text)
    assert "info.json episodes: 5" in text
    if PRE_FIX:
        assert "info.json frames" not in text, "pre-fix: info.json frames line hidden"
        print(">>> PRE-FIX CONFIRMED: 'info.json frames' line is never rendered.\n")
    else:
        assert "info.json frames  : 450" in text, \
            "post-fix: info.json frames: 450 must appear beside info.json episodes"
        print(">>> FIX OK: 'info.json frames: 450' is now rendered beside its sibling.\n")


def case_3_asymmetry_on_pass() -> None:
    """Sibling pair was asymmetric even on PASS - docs/learn/data/verify.md:38
    reflects exactly that gap."""
    root = pathlib.Path("/tmp/verify_fmt_case3")
    _mk_dataset(root, episodes=[(0, 150), (1, 150), (2, 150)],
                info_total_episodes=3, info_total_frames=450)
    text = _run_cli(root, "--expected", "3",
                    "--no-check-videos", "--no-check-stats")
    print("=== Case 3: PASS case still shows asymmetric pair (pre-fix) ===")
    print(text)
    assert "[PASS]" in text
    assert "info.json episodes: 3" in text
    if PRE_FIX:
        assert "info.json frames" not in text
        print(">>> PRE-FIX CONFIRMED: PASS output omits info.json frames.\n")
    else:
        assert "info.json frames  : 450" in text
        print(">>> FIX OK: PASS output renders the full declared pair.\n")


if __name__ == "__main__":
    try:
        case_1_zero_frames_mega_episode()
        case_2_info_frames_rendering()
        case_3_asymmetry_on_pass()
    except AssertionError as e:
        print(f"ASSERTION FAILED: {e}")
        sys.exit(2)
    label = "PRE-FIX bug reproduced" if PRE_FIX else "FIX verified"
    print(f"{label}: three asymmetries in verify_dataset._format_report "
          "(strands_robots/verify_dataset.py:704-723).")
