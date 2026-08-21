"""Is there room on disk for the episodes you are about to record? (BUGS.md Q92)

Nothing in this dashboard has ever looked at free space. Recording writes parquet shards and
per-camera video into the lerobot dataset home for as long as the operator keeps pressing "record",
and the failure mode when the volume fills is the worst kind: the session reports success episode by
episode, a shard write fails somewhere inside lerobot, and what is left on disk is a dataset whose
``meta/info.json`` promises episodes its ``data/`` cannot supply. The operator finds out at TRAIN
time, hours later, and the error will name a parquet file rather than the disk.

Measured on cagatay's Mac mini the day this was written (Q91): 18.0Gi free, falling ~2Gi/h because
macOS was allocating a 1Gi swapfile every ~25 minutes for two 6Gi processes. A record session
started that evening would have hit a full volume mid-run with nothing on screen having mentioned it.

DESIGN, and the part that matters: this WARNS, it does not refuse.
* An estimate cannot refuse. Bytes per episode depends on the cameras, their resolution, the fps and
  how long the operator holds the arm, and this module is given none of that for a first recording.
  A refusal built on a guess would stop legitimate work; a warning built on a guess costs a glance.
* The one exception is a volume with so little left that the write cannot plausibly finish
  (``CRITICAL_MB``) -- there the refusal is not a guess about the dataset, it is a statement about
  the disk, and it still names the number so the operator can disagree with it.
* An UNREADABLE free-space reading produces no verdict at all. Same law as every other check here:
  no evidence must never be able to block a recording.

Everything except :func:`free_space` is pure.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

MB = 1024 * 1024
GB = 1024 * MB

#: Below this, the disk itself is the story: refuse and say so.
CRITICAL_MB = 2 * 1024
#: Below this, warn. Two hours of the swap growth measured in Q91, and comfortably more than a
#: single multi-camera episode - the point is to be told BEFORE a long session, not after one.
TIGHT_MB = 12 * 1024


def _fmt(mb: float) -> str:
    """Sizes the way an operator reads them: 18Gi, 940Mi."""
    return f"{mb / 1024:.1f}Gi" if mb >= 1024 else f"{mb:.0f}Mi"


def headroom_verdict(
    *,
    free_mb: float | None,
    total_mb: float | None = None,
    where: str | None = None,
) -> dict[str, Any] | None:
    """Classify the space where a dataset is about to be written.

    Returns ``None`` for no reading and for a comfortable one -- a check that speaks when there is
    nothing to say trains the operator to dismiss it. Otherwise a dict with:
      ``level``    'critical' (a refusal is justified) or 'tight' (a warning);
      ``headline`` the fact, with the number in it;
      ``advice``   what to do, naming that datasets and swap share this volume.
    """
    if free_mb is None or free_mb < 0:
        return None
    free = float(free_mb)
    if free >= TIGHT_MB:
        return None
    place = f" on {where}" if where else ""
    share = ""
    if total_mb and total_mb > 0:
        share = f" ({free / float(total_mb) * 100:.1f}% of the volume)"
    if free < CRITICAL_MB:
        return {
            "level": "critical",
            "free_mb": round(free),
            "headline": f"only {_fmt(free)} free{place}{share} - not enough to finish a recording",
            "advice": (
                "A dataset write that runs out of space leaves meta/info.json promising episodes "
                "that data/ cannot supply, and the error surfaces at TRAIN time naming a parquet "
                "file. Free space first: this volume also holds macOS swap, so a memory-hungry "
                "process costs disk too."
            ),
        }
    return {
        "level": "tight",
        "free_mb": round(free),
        "headline": f"{_fmt(free)} free{place}{share} - fine for a short session, tight for a long one",
        "advice": (
            "Cameras dominate the size: a multi-camera episode is far bigger than the joint stream. "
            "Watch the number if you plan many episodes - and remember this volume holds macOS swap, "
            "so free space can fall while you record even if nothing else is writing."
        ),
    }


def free_space(path: str | Path | None = None) -> dict[str, Any]:
    """Read free space where datasets land. Never raises.

    Defaults to the lerobot dataset home and walks UP to the first existing ancestor: a dataset home
    that does not exist yet still sits on a volume, and the answer for that volume is the answer.
    """
    try:
        target = Path(path) if path is not None else _dataset_home()
        # A RELATIVE path would walk up to "." and report the volume this process happens to be
        # running in - a reading for a disk nobody asked about, presented as the dataset's disk.
        # Caught by the test that passed a nonsense path and got a confident number for the CWD.
        if not target.is_absolute():
            return {}
        probe = target
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        usage = shutil.disk_usage(probe)
        return {
            "path": str(target),
            "measured_at": str(probe),
            "free_mb": usage.free / MB,
            "total_mb": usage.total / MB,
        }
    except Exception:  # noqa: BLE001 - no reading is a valid answer, an exception is not
        return {}


def _dataset_home() -> Path:
    try:
        from strands_robots.dataset_recorder import resolve_dataset_dir

        return resolve_dataset_dir("local/_headroom_probe").parent
    except Exception:  # noqa: BLE001
        return Path.home() / ".cache" / "huggingface" / "lerobot"
