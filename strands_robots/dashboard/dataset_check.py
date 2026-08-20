"""Is a listed dataset a dataset, or the folder an aborted recording left behind? (BUGS.md Q37)

``training.local_datasets`` calls a directory a dataset when ``meta/info.json`` exists -- the same
"discovered by its config" shape as Q36. And ``DatasetRecorder.create`` calls
``LeRobotDataset.create()`` when a recording session OPENS, which writes ``meta/info.json``
immediately, before a single episode is captured. So a session that was opened and then abandoned
(the operator closed the sheet, the arms failed to connect, the process died) leaves a directory
that lists as a dataset for ever after.

Measured on this machine: ``local/sim_recording`` reports ``total_episodes: 0``, has no ``data/``
directory at all, and the training tab offers it with a "replay in sim" button next to it. Picking
it costs a training run that fails minutes in, or a replay that shows nothing.

The verdict is computed from the metadata the listing already reads plus one directory probe, so
listing 50 datasets does not become 50 dataset loads. It never opens a parquet file, and its
wording says so: a dataset can be unusable in ways only a loader can see (a schema mismatch, a
corrupt shard), and this check will pass those.

What it must NOT do: compare the number of data FILES to the number of episodes. Measured on the
same machine: a healthy 30-episode v3.0 dataset holds exactly ONE parquet file, because v3 packs
many episodes per file (``data/chunk-000/file-000.parquet``). A count comparison would condemn
every correct v3 dataset on disk.
"""

from __future__ import annotations

from typing import Any, Mapping

__all__ = ["dataset_verdict", "MIN_EPISODES"]

#: One episode is a real dataset (a single demonstration you can replay or overfit on). Zero is
#: not "small", it is "nothing was ever recorded".
MIN_EPISODES = 1


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def dataset_verdict(
    meta: Mapping[str, Any] | None,
    *,
    has_data_files: bool | None = None,
) -> dict[str, Any]:
    """What can be said about a listed dataset without loading it.

    ``meta`` is the parsed ``meta/info.json`` fields the listing already keeps
    (``total_episodes``, ``total_frames``, ``fps``, ``robot_type``); an empty/None mapping means
    the file could not be read. ``has_data_files`` is one directory probe by the caller -- ``None``
    when it did not look, which is honestly reported as "not checked" rather than assumed either
    way.

    Returns ``{"usable": bool, ...}``. ``usable`` means "nothing visible from the metadata says
    this is empty or missing its frames" -- never "this loads".
    """
    m = dict(meta or {})
    episodes = _as_int(m.get("total_episodes"))
    frames = _as_int(m.get("total_frames"))

    if not m or (episodes is None and frames is None and m.get("fps") is None):
        return {
            "usable": False,
            "reason": "unreadable_meta",
            "problem": (
                "meta/info.json could not be read, so nothing is known about this directory's "
                "contents. If a recording is running right now, wait for it to finish; otherwise "
                "the file is truncated and the recording did not survive."
            ),
        }

    if episodes is not None and episodes < MIN_EPISODES:
        return {
            "usable": False,
            "reason": "no_episodes",
            "episodes": episodes,
            "problem": (
                "0 episodes. meta/info.json is written when a recording session OPENS, before the "
                "first episode is captured, so a directory like this is what an abandoned session "
                "leaves behind - not a dataset. Record into it, or delete it."
            ),
        }

    if has_data_files is False:
        return {
            "usable": False,
            "reason": "missing_data",
            "episodes": episodes,
            "problem": (
                f"meta/info.json claims {episodes if episodes is not None else 'some'} episodes but "
                "there are no data files under data/. The metadata was written and the frames never "
                "landed (a crash mid-recording), or the data directory was moved or deleted while "
                "meta/ stayed behind. Training would fail after the setup it charges you for."
            ),
        }

    verdict: dict[str, Any] = {
        "usable": True,
        "episodes": episodes,
        "note": "read from meta/info.json only - nothing here opens a shard",
    }
    if frames == 0 and episodes:
        # Episodes recorded, no frames counted: legal-but-odd metadata rather than a refusal,
        # because the frame count is a summary and the episodes are the substance.
        verdict["warning"] = (
            f"{episodes} episode(s) but 0 frames counted in the metadata - the summary may be "
            "stale; check the episode list before a long training run."
        )
    if has_data_files is None:
        verdict["note"] += "; data/ was not checked"
    return verdict


#: A row that a recorder is writing into right now. NOT the same claim as "broken": the folder is
#: mid-flight, and the advice for an abandoned one ("delete it") would destroy a session in
#: progress. Reported as its own reason so the UI can say so.
RECORDING_REASON = "recording_in_progress"


def _same_dataset(row: Mapping[str, Any], active: str) -> bool:
    """Is this listing row the dataset the recorder named?

    A recorder names a ``repo_id`` ("local/sim_recording"); a listing row carries both a
    ``repo_id`` (relative to whichever root the scan walked) and an absolute ``root``. The two can
    disagree - a dataset discovered under a remembered collect root lists as "sim_recording" while
    the recorder calls it "local/sim_recording" - so a repo_id-only comparison would silently miss
    the very session it exists to notice. The path tail is checked as well, with a separator
    guard so "local/sim_recording" cannot match ".../not_sim_recording".
    """
    if not active:
        return False
    repo = str(row.get("repo_id") or "")
    if repo and repo == active:
        return True
    root = str(row.get("root") or "")
    if root:
        return root == active or root.endswith("/" + active.strip("/"))
    return False


def mark_live_recording(
    rows: list[dict[str, Any]],
    active_dataset: str | None,
    *,
    episodes_so_far: int | None = None,
) -> list[dict[str, Any]]:
    """Re-judge the row a recording session is writing into (Q38).

    A dataset in mid-recording looks EXACTLY like an abandoned one to a metadata check: episode 0
    is not in ``meta/info.json`` until it is flushed, so a session that opened a minute ago reads
    as "0 episodes - an abandoned session, record into it or delete it". That advice is not merely
    unhelpful, it names the one action that would destroy the recording in progress.

    Rows are returned as new dicts (the caller's cache is not mutated), and only the matching row
    is touched. ``usable`` stays False for it: training would read a dataset that is still growing
    and a replay would race the writer - but the REASON, and therefore the sentence the operator
    reads, is now the true one.
    """
    if not active_dataset:
        return rows
    out: list[dict[str, Any]] = []
    for row in rows:
        if not _same_dataset(row, active_dataset):
            out.append(row)
            continue
        n = episodes_so_far if isinstance(episodes_so_far, int) else None
        so_far = f"{n} episode(s) captured so far" if n is not None else "episodes are being written"
        out.append({
            **row,
            "usable": False,
            "recording": True,
            "reason": RECORDING_REASON,
            "problem": (
                f"a recording session is writing into this dataset right now - {so_far}. "
                "Training would read a dataset that is still growing, and a replay would race the "
                "writer. Wait for the session to close; do NOT delete the folder."
            ),
        })
    return out
