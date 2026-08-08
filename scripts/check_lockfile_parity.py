#!/usr/bin/env python3
"""Refuse a lockfile that no longer resolves the manifest.

``uv.lock`` is checked in, so it *looks* authoritative: it pins the exact version
of every package a locked install resolves. Nothing compared it against
``pyproject.toml``, and the two drifted apart for two weeks with no signal. A
stale lock is not cosmetic -- it can pin a version the manifest forbids (defeating
a floor a change was made specifically to buy), or omit a dependency an extra
declares (so the extra resolves without the module its own code imports).

Two independent checks, because they fail for different reasons:

1. **Manifest parity** -- ``uv.lock`` records its own transcription of the
   manifest in the root package's ``[package.metadata] requires-dist``. That
   transcription must equal what ``pyproject.toml`` declares today. A difference
   means the lock was written against a *different* manifest, which is what "the
   lock no longer resolves the manifest" is. This catches a dependency added to an
   extra and never locked.

2. **Pin satisfaction** -- the version the lock pins must satisfy the specifier
   the manifest declares. Parity alone would pass a lock whose transcription was
   refreshed while its resolution was not, and this is the check whose failure
   names the consequence a caller actually experiences: the pin violates its own
   floor.

Both are offline and deterministic -- they read the two files and compare them,
with no resolver, no network and no ``uv`` binary. That is what lets the guard run
as an ordinary test in the required suite. ``uv lock --check`` is the
authoritative equivalent and stays the way to *repair* a finding, but it has to
resolve, so it cannot be a unit test.

Two encodings this mirrors rather than approximates, both pinned by the
accompanying tests:

* **Self-references are expanded.** The manifest builds its extras out of
  self-references -- ``[all]`` declares ``strands-robots[rosbridge]`` rather than
  repeating that extra's requirements -- and uv resolves those away, so the
  recorded set is the transitive closure. Comparing the literal declared list
  would report every inherited requirement as unlocked.
* **A name may be locked more than once.** ``[tool.uv] conflicts`` forks the
  resolution, so a handful of names appear twice at different versions. A declared
  requirement is satisfied when *some* locked version satisfies it; demanding it
  of every fork would fail the conflict declaration that keeps the forks apart.

Exit code is 0 when the audit passes and 1 when any finding is reported, so the
script can gate directly. Run it after editing ``pyproject.toml``:

    python3 scripts/check_lockfile_parity.py
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

from packaging.markers import InvalidMarker, Marker
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

# uv renders an extra-gated requirement's marker with this clause. Whatever else
# the marker holds is compared as a residual, so a requirement that later grows a
# platform marker is still compared rather than silently dropped.
_EXTRA_CLAUSE_RE = re.compile(r"extra == '([^']+)'")

# (extra or None for a core dependency, name, extras, specifiers, residual marker)
RequirementKey = tuple[str | None, str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]


def canonical_name(name: str) -> str:
    """Return the PEP 503 canonical (lowercased, dash-normalized) name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def marker_clauses(marker: str | None) -> tuple[str, ...]:
    """Return a marker's top-level ``and`` clauses, normalized and sorted.

    Sorted because uv is free to order the clauses differently from the manifest,
    and normalized through :class:`~packaging.markers.Marker` so a manifest
    written with double quotes compares equal to uv's single quotes. A fragment
    that does not parse on its own (a parenthesized ``or`` group) is kept verbatim
    rather than dropped, so an unmodelled marker shape can only ever produce a
    finding, never a silent pass.
    """
    if not marker:
        return ()
    clauses: list[str] = []
    for fragment in marker.split(" and "):
        clause = fragment.strip()
        if not clause:
            continue
        try:
            clauses.append(str(Marker(clause)))
        except InvalidMarker:
            clauses.append(clause)
    return tuple(sorted(clauses))


def _render(name: str, extras: tuple[str, ...], specifiers: tuple[str, ...]) -> str:
    suffix = f"[{','.join(extras)}]" if extras else ""
    return f"{name}{suffix}{','.join(specifiers)}"


def declared_requirements(pyproject_path: Path) -> dict[RequirementKey, str]:
    """Return {key: human-readable requirement} the manifest declares.

    A self-reference to another of the project's own extras (``[sim-mujoco]``
    declares ``strands-robots[sim]``, for one) is expanded transitively, because
    that is what uv records. A cycle in the extras graph terminates rather than
    recursing forever.
    """
    project = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))["project"]
    own = canonical_name(project.get("name", ""))
    extras = {canonical_name(k): v for k, v in project.get("optional-dependencies", {}).items()}

    def expand(extra: str, seen: frozenset[str]) -> list[str]:
        if extra in seen:
            return []
        collected: list[str] = []
        for raw in extras.get(extra, []):
            requirement = Requirement(raw)
            if own and canonical_name(requirement.name) == own:
                for inherited in requirement.extras:
                    collected.extend(expand(canonical_name(inherited), seen | {extra}))
            else:
                collected.append(raw)
        return collected

    def key(extra: str | None, requirement: Requirement) -> RequirementKey:
        return (
            extra,
            canonical_name(requirement.name),
            tuple(sorted(canonical_name(e) for e in requirement.extras)),
            tuple(sorted(str(s) for s in requirement.specifier)),
            marker_clauses(str(requirement.marker) if requirement.marker else None),
        )

    found: dict[RequirementKey, str] = {}
    for raw in project.get("dependencies", []):
        found[key(None, Requirement(raw))] = f"[core] {raw}"
    for extra in extras:
        for raw in expand(extra, frozenset()):
            found[key(extra, Requirement(raw))] = f"[{extra}] {raw}"
    return found


def recorded_requirements(lock_path: Path, project_name: str) -> dict[RequirementKey, str]:
    """Return {key: human-readable requirement} the lock records for the project.

    Reads the root package's ``[package.metadata] requires-dist``, which is uv's
    own transcription of the manifest it last resolved.
    """
    own = canonical_name(project_name)
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    found: dict[RequirementKey, str] = {}
    for package in lock.get("package", []):
        if canonical_name(package.get("name", "")) != own:
            continue
        for entry in package.get("metadata", {}).get("requires-dist", []):
            marker = entry.get("marker") or ""
            match = _EXTRA_CLAUSE_RE.search(marker)
            extra = canonical_name(match.group(1)) if match else None
            residual = _EXTRA_CLAUSE_RE.sub("", marker).strip()
            residual = re.sub(r"^and\s+|\s+and$", "", residual).strip()
            name = canonical_name(entry["name"])
            entry_extras = tuple(sorted(canonical_name(e) for e in entry.get("extras", [])))
            specifiers = tuple(sorted(str(s) for s in SpecifierSet(entry.get("specifier", ""))))
            key: RequirementKey = (extra, name, entry_extras, specifiers, marker_clauses(residual or None))
            found[key] = f"[{extra or 'core'}] {_render(name, entry_extras, specifiers)}"
    return found


def locked_versions(lock_path: Path) -> dict[str, list[str]]:
    """Return {canonical name: every version the lock pins for it}.

    A list rather than one version because ``[tool.uv] conflicts`` forks the
    resolution, so one name can be locked at several versions.
    """
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    versions: dict[str, list[str]] = {}
    for package in lock.get("package", []):
        version = package.get("version")
        if version:
            versions.setdefault(canonical_name(package["name"]), []).append(str(version))
    return versions


def parity_findings(declared: dict[RequirementKey, str], recorded: dict[RequirementKey, str]) -> list[str]:
    """Return one finding per requirement the manifest and the lock disagree on."""
    findings = [
        f"NOT LOCKED {declared[key]}: pyproject declares it, uv.lock does not record it"
        for key in sorted(declared.keys() - recorded.keys(), key=repr)
    ]
    findings += [
        f"STALE LOCK {recorded[key]}: uv.lock records it, pyproject no longer declares it"
        for key in sorted(recorded.keys() - declared.keys(), key=repr)
    ]
    return findings


def pin_findings(declared: dict[RequirementKey, str], versions: dict[str, list[str]]) -> list[str]:
    """Return one finding per declared requirement whose locked pin violates it.

    A requirement whose package is absent from the lock is left to
    :func:`parity_findings`, which names that case directly.
    """
    findings: list[str] = []
    for key, described in sorted(declared.items(), key=lambda item: repr(item[0])):
        _, name, _, specifiers, _ = key
        pinned = versions.get(name)
        if not pinned or not specifiers:
            continue
        specifier = SpecifierSet(",".join(specifiers))
        if not any(specifier.contains(version, prereleases=True) for version in pinned):
            findings.append(
                f"PIN VIOLATES FLOOR {described}: uv.lock pins {name} "
                f"{' / '.join(sorted(pinned))}, which does not satisfy {specifier}"
            )
    return findings


def audit(pyproject_path: Path, lock_path: Path) -> list[str]:
    """Return every finding for the manifest/lock pair (empty when they agree)."""
    project_name = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))["project"]["name"]
    declared = declared_requirements(pyproject_path)
    recorded = recorded_requirements(lock_path, project_name)
    return parity_findings(declared, recorded) + pin_findings(declared, locked_versions(lock_path))


def main() -> int:
    """Report every finding; return 1 when the lock does not resolve the manifest."""
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Refuse a lockfile that does not resolve the manifest.")
    parser.add_argument("--pyproject", type=Path, default=repo_root / "pyproject.toml")
    parser.add_argument("--lock", type=Path, default=repo_root / "uv.lock")
    args = parser.parse_args()

    findings = audit(args.pyproject, args.lock)
    if not findings:
        print(f"OK: {args.lock.name} resolves {args.pyproject.name}")
        return 0
    print(f"{len(findings)} finding(s): {args.lock.name} does not resolve {args.pyproject.name}\n")
    for finding in findings:
        print(f"  {finding}")
    print("\nRun `uv lock` to bring the lockfile back in line with the manifest.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
