"""Distributions an install of one extra resolves to, read from ``uv.lock``.

Several packaging rules ask the same question - does an install of this extra
really ship the dependency the manifest claims for it - and each answered it with
its own copy of the same walk. One owner, because a copy that drifts answers a
different question under the same name.

The walk reads the checked-in lock rather than resolving live, so a rule that
grades the manifest stays offline and deterministic. Markers are deliberately not
evaluated: the result is a superset of what any one platform installs, which is
the safe direction for a rule that fails a manifest - it never reports a
distribution absent that some environment does resolve.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.utils import canonicalize_name

_LOCK = Path(__file__).resolve().parents[1] / "uv.lock"

#: The project's own name in the lock, and the root every walk starts from.
ROOT = "strands-robots"


def locked_packages() -> dict[str, list[dict]]:
    """Every ``[[package]]`` entry in the lock, grouped by distribution name.

    A name can carry several entries - uv records one per resolution fork (the
    platform markers on ``torchcodec`` produce two entries) - so the value is a
    list and the walk reads all of them.
    """
    with _LOCK.open("rb") as handle:
        locked = tomllib.load(handle)["package"]
    by_name: dict[str, list[dict]] = {}
    for package in locked:
        by_name.setdefault(package["name"], []).append(package)
    return by_name


def requested_extras(entry: dict) -> tuple[str, ...]:
    """Extras a lock dependency entry asks for. uv spells the key ``extra``.

    Singular, and easy to mistake for ``extras``: reading the plural silently
    drops every ``package[extra]`` requirement from the walk, which reports a
    transitively supplied distribution as absent.
    """
    requested = entry.get("extra", ())
    return tuple(requested) if isinstance(requested, list) else (requested,)


def lock_closure(extra: str | None) -> frozenset[str]:
    """Canonical names an install of *extra* resolves to (``None`` = base install).

    Args:
        extra: One entry of ``[project.optional-dependencies]``, or ``None`` for
            the unconditional ``[project.dependencies]`` alone.

    Returns:
        Every distribution reachable from that root, canonicalized, including the
        root itself.
    """
    by_name = locked_packages()
    root_extras: tuple[str, ...] = () if extra is None else (extra,)
    seen: set[tuple[str, tuple[str, ...]]] = set()
    pending: list[tuple[str, tuple[str, ...]]] = [(ROOT, root_extras)]
    while pending:
        name, extras = pending.pop()
        if (name, extras) in seen:
            continue
        seen.add((name, extras))
        for package in by_name.get(name, []):
            entries = list(package.get("dependencies", []))
            optional = package.get("optional-dependencies", {})
            for group in extras:
                entries += optional.get(group, [])
            pending += [(entry["name"], requested_extras(entry)) for entry in entries]
    return frozenset(str(canonicalize_name(name)) for name, _ in seen)
