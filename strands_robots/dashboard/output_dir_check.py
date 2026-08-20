"""What a training run is about to DO to the directory you typed (Q58).

The training form's ``output_dir`` is free text with placeholder ``/tmp/my_policy_ckpt``, and
``LerobotTrainer.start()`` contains this fresh-start hygiene::

    if not spec.resume and os.path.isdir(spec.output_dir):
        if self.latest_checkpoint(spec.output_dir) is None:
            shutil.rmtree(spec.output_dir, ignore_errors=True)

So typing a path that already exists and holds no resumable checkpoint makes the dashboard
**recursively delete that directory**, with nothing on screen saying so and no way to undo it. A
mistyped or reused path — a dataset dir, a notes folder, a previous run whose checkpoint was already
exported and moved away — is wiped by pressing "train".

The opposite case fails differently and just as quietly: a directory that DOES hold a checkpoint is
kept, and lerobot's own validate then refuses a non-resume run into an existing output_dir. The
dashboard cannot ask for a resume at all (``resume`` is not in ``SPEC_KEYS``), and by then submit()
has already returned success with a job id, so the refusal only exists in the run's log file.

This module answers the question BEFORE the run: what is in there, and what will happen to it.
Everything except :func:`inspect_output_dir` is pure.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

#: How many names to show. Enough to recognise a directory, few enough to read.
SAMPLE = 5


def classify_output_dir(
    *,
    exists: bool,
    is_dir: bool = True,
    has_checkpoint: bool = False,
    names: list[str] | None = None,
    total: int = 0,
    unreadable: str | None = None,
) -> dict[str, Any]:
    """Verdict for one candidate ``output_dir``.

    ``state`` is one of:

    ``free``
        nothing there (or an empty directory) — the run creates it.
    ``resumable``
        a checkpoint lives there. NOT destructive, but this dashboard cannot resume, so the run
        will be refused by lerobot itself. Needs a different directory, not a confirmation.
    ``occupied``
        real files with no checkpoint — pressing train DELETES them. ``destructive`` is True and
        ``needs_confirm`` is True: the operator must say yes to a named loss, not to a shrug.
    ``not_a_dir``
        the path is a FILE. Refused outright rather than confirmed: nothing in the training flow
        wants to write a run into a file's name, so this is a typo, not a decision.
    ``unknown``
        the path could not be read. Never treated as ``free`` — "I could not look" and "there is
        nothing there" lead to opposite advice, and guessing the friendly one here is guessing
        about a delete.
    """
    if unreadable:
        return {
            "state": "unknown",
            "destructive": False,
            "needs_confirm": False,
            "detail": (
                f"cannot read that path ({unreadable}), so what a run would do to it is unknown — "
                "fix the path or the permissions before training"
            ),
        }
    if not exists:
        return {
            "state": "free",
            "destructive": False,
            "needs_confirm": False,
            "detail": "does not exist yet — the run creates it",
        }
    if not is_dir:
        return {
            "state": "not_a_dir",
            "destructive": False,
            "needs_confirm": False,
            "detail": "that path is a FILE, not a directory — a run cannot be written there",
        }
    if has_checkpoint:
        return {
            "state": "resumable",
            "destructive": False,
            "needs_confirm": False,
            "detail": (
                "already holds a training checkpoint. It will NOT be deleted, but this dashboard "
                "cannot resume a run, and lerobot refuses a fresh run into an existing run's "
                "directory — pick a new directory (the failure would otherwise appear only in "
                "the run's log, after the job says it started)"
            ),
        }
    shown = list(names or [])[:SAMPLE]
    if not shown and total == 0:
        return {
            "state": "free",
            "destructive": False,
            "needs_confirm": False,
            "detail": "exists but is empty — the run uses it as it is",
        }
    more = max(0, total - len(shown))
    listing = ", ".join(shown) + (f" and {more} more" if more else "")
    return {
        "state": "occupied",
        "destructive": True,
        "needs_confirm": True,
        "entries": shown,
        "total": total,
        "detail": (
            f"holds {total} item(s) and NO training checkpoint, so starting a run here "
            f"DELETES the directory and everything in it ({listing}). This cannot be undone"
        ),
    }


def inspect_output_dir(path: str, *, has_checkpoint: Any = None) -> dict[str, Any]:
    """Read the path and classify it. ``has_checkpoint`` may be a callable(path) -> bool.

    The checkpoint test is injected because the authority on "is this resumable" is the trainer
    (``LerobotTrainer.latest_checkpoint``, which recognises a checkpoint BY ITS CONFIG FILE) — this
    module must not grow a second, disagreeing definition of the thing that decides whether a
    delete happens.
    """
    p = Path(path).expanduser()
    verdict_path = str(p)
    try:
        if not p.exists():
            out = classify_output_dir(exists=False)
            out["path"] = verdict_path
            return out
        if not p.is_dir():
            out = classify_output_dir(exists=True, is_dir=False)
            out["path"] = verdict_path
            return out
        entries = sorted(e.name for e in os.scandir(p))
        ckpt = False
        if has_checkpoint is not None:
            try:
                ckpt = bool(has_checkpoint(verdict_path))
            except Exception:  # noqa: BLE001 - a trainer hiccup must not silently mean "no ckpt"
                out = classify_output_dir(exists=True, unreadable="checkpoint probe failed")
                out["path"] = verdict_path
                return out
        out = classify_output_dir(
            exists=True, has_checkpoint=ckpt, names=entries, total=len(entries)
        )
    except OSError as exc:
        out = classify_output_dir(exists=True, unreadable=type(exc).__name__)
    out["path"] = verdict_path
    return out


def default_checkpoint_probe(path: str) -> bool:
    """The trainer's own answer, when it can be imported (torch-less installs say False)."""
    try:
        from strands_robots.training.lerobot import LerobotTrainer

        return LerobotTrainer.latest_checkpoint(LerobotTrainer.__new__(LerobotTrainer), path) is not None
    except Exception:  # noqa: BLE001
        # Fall back to the on-disk shape lerobot writes: <output_dir>/checkpoints/<step>/…
        return Path(path, "checkpoints").is_dir()
