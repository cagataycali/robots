"""What build this dashboard process actually is — the one thing /api/health never said.

Every diagnosis in this project has had to answer "is the running server older than the shipped
bundle?" by hand: read the process start time, then guess which features that predates. Four days
of notes say "waits for a terminal-started restart" because that guess is the only tool there was.
A build stamp turns it into one request.

Two design choices worth stating, because they are what makes the stamp trustworthy:

* NO SUBPROCESS. Shelling out to `git` inside a request is a fork per poll, it inherits whatever
  PATH the daemon had, and it fails in a way that looks like a broken endpoint. The refs are plain
  files; reading them is a few bytes and cannot hang.
* The stamp is ALWAYS present, unlike this endpoint's news-only blocks (joint_streams,
  refused_handshakes). That is deliberate: absence is the signal. A server that answers /api/health
  with no `build` key is, by construction, older than the commit that added it — which is exactly
  the question a UI needs answered when a field it renders is missing.

Unknowns are reported as None rather than guessed strings: "unknown commit" and "commit 0000000"
both read like data, and a wrong build id is worse than a missing one.
"""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Any


def read_commit(root: Path | str | None) -> str | None:
    """The short sha of ``root``'s checkout, or None when it cannot be read honestly.

    Handles the two shapes a HEAD file takes: a symbolic ref into refs/heads (a branch checkout,
    which is how this repo is always used) and a detached HEAD holding the sha directly. A packed
    ref is NOT chased - if refs/heads/<branch> is absent the answer is None, because a stamp that
    silently reports the wrong commit is worse than one that admits ignorance.
    """
    if root is None:
        return None
    git = Path(root) / ".git"
    try:
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None
    if head.startswith("ref:"):
        ref = head[4:].strip()
        # A ref path is attacker-irrelevant here (it is our own checkout) but still must not escape
        # .git: a HEAD containing "ref: ../../etc/passwd" should yield None, not a file read.
        if ".." in Path(ref).parts or Path(ref).is_absolute():
            return None
        try:
            head = (git / ref).read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            return None
    head = head.split()[0] if head else ""
    if len(head) < 7 or any(c not in "0123456789abcdef" for c in head.lower()):
        return None
    return head[:12]


def package_version() -> str | None:
    """The installed strands-robots version, or None in a source checkout with nothing installed."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except Exception:  # pragma: no cover - stdlib always has it on 3.12
        return None
    for name in ("strands-robots", "strands_robots"):
        try:
            return version(name)
        except PackageNotFoundError:
            continue
        except Exception:  # pragma: no cover - a broken dist must not break /api/health
            return None
    return None


def stamp(*, commit: str | None, version: str | None, started: float) -> dict[str, Any]:
    """Assemble the payload. Pure, so the shape is testable without a filesystem or a clock."""
    return {"commit": commit, "version": version, "started": started}


@lru_cache(maxsize=1)
def build_info(root: str | None = None) -> dict[str, Any]:
    """Cached stamp for this process. Cached because it cannot change without a restart -
    that immutability IS the fact being reported, and /api/health is polled constantly."""
    here = Path(__file__).resolve()
    # strands_robots/dashboard/build_info.py -> repo root is three parents up.
    default_root = here.parents[2]
    return stamp(
        commit=read_commit(root if root is not None else default_root),
        version=package_version(),
        started=time.time(),
    )
