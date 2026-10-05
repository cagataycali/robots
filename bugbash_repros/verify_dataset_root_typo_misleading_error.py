"""
Minimal repro for Error-UX defect: `verify_dataset` collapses four distinct
user-mistake shapes onto one message that only fits ONE of them.

What a working robotics engineer actually hits:
    1. typo'd path (dir does not exist)
    2. forgot to append the dataset name (points at a file)
    3. broken symlink (post-rsync fallout)
    4. honestly empty dataset-root (unfinalized recording)

All four yield identical output:
    > No meta/episodes parquet under <path>. The dataset is empty or was
    > never finalized (episodes are flushed to parquet at
    > stop_recording/finalize).

Only case (4) matches the fix the message prescribes. The other three send
a user hunting for a stop_recording() call that was never missing.

Reproduces on strands-labs/robots main @ a0693e4 (v0.5.3 target).

Fix sketch (14 LOC in dataset_metadata.py `read_dataset_episode_indices`):

    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root {root_path} does not exist (typo?).")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Dataset root {root_path} is not a directory.")
    # ... existing FileNotFoundError, now reached only for the real case
"""
import pathlib

from strands_robots.verify_dataset import verify_dataset


def one_message_for_four_mistakes() -> None:
    cases = {
        "typo (dir does not exist)":        pathlib.Path("/tmp/strands_bugbash_does_not_exist"),
        "root is a regular file":           pathlib.Path("/tmp/strands_bugbash_file"),
        "root is a broken symlink":         pathlib.Path("/tmp/strands_bugbash_broken_link"),
        "honestly empty (unfinalized)":     pathlib.Path("/tmp/strands_bugbash_empty_dir"),
    }
    # Clean prior staging so the repro is deterministic.
    for p in cases.values():
        if p.is_symlink() or p.exists():
            if p.is_dir() and not p.is_symlink():
                for child in p.iterdir():
                    child.unlink()
                p.rmdir()
            else:
                p.unlink()
    # Stage the four physical shapes.
    cases["root is a regular file"].write_text("")
    cases["root is a broken symlink"].symlink_to("/nope_definitely_does_not_exist")
    cases["honestly empty (unfinalized)"].mkdir(exist_ok=True)

    for label, root in cases.items():
        rep = verify_dataset(root, expected=5)
        msg = rep["problems"][0] if rep["problems"] else "<no problem>"
        print(f"[{label}]")
        print(f"  status  : {rep['status']}")
        print(f"  problem : {msg}")
        print()


if __name__ == "__main__":
    one_message_for_four_mistakes()
