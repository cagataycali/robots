#!/usr/bin/env python3
"""Assemble ``changelog.d/`` news fragments into ``CHANGELOG.md``.

Why this exists
---------------
Every behavioural PR used to append its entry directly to the top of the
``## [Unreleased]`` section of ``CHANGELOG.md``. Because every branch inserts at
the *same anchor*, any two PRs open at once are guaranteed to conflict the
moment either one merges -- on ordering alone, never on meaning. The repository
dismisses stale approvals on push, so clearing a conflict that is purely an
append-ordering artifact costs a re-approval round on every affected PR, and the
cost scales as O(open PRs). One recorded instance: a single merge to ``main``
turned five already-approved PRs ``CONFLICTING``, with ``CHANGELOG.md`` as the
only conflicting file in all five.

A ``merge=union`` attribute does not fix that: it removes the manual resolution
step, but GitHub's mergeability computation does not apply merge drivers, so the
PR still reports ``CONFLICTING``, the merge button stays disabled, and the
resolving push still dismisses the approval.

A news fragment makes the conflict *structurally impossible*: each PR adds its
own file under ``changelog.d/``, so no two PRs ever touch the same path and
there is nothing for a merge to reconcile. This script is the release-time
assembly step that pays for that: it validates every accumulated fragment and
folds them into the ``[Unreleased]`` section of ``CHANGELOG.md``.

Fragment contract
-----------------
- Path: ``changelog.d/<number>-<slug>.md``, where ``<number>`` is the PR (or
  issue) number and ``<slug>`` is lowercase ``a-z0-9`` words joined by ``-``.
- Content: exactly the Markdown that would have been pasted into
  ``CHANGELOG.md`` -- one ``### <Category>: <summary>`` heading followed by the
  prose body, in the style of the entries already in the log.
- A fragment may not contain a level-2 (``## ``) heading: that would forge a
  version section and break the structural contract pinned by
  ``tests/test_changelog_format.py``.

Anything else in ``changelog.d/`` (a stray ``.txt``, a misnamed ``.md``) is a
hard error rather than a skipped file -- a fragment that is silently ignored is
a behavioural change that never reaches the log.

Usage
-----
``--check``   validate the fragments and exit non-zero on any problem
              (safe to run in CI; writes nothing).
``--print``   write the assembled ``[Unreleased]`` body to stdout.
``--apply``   fold the fragments into ``CHANGELOG.md`` and delete the consumed
              files. Refuses wholesale if *any* fragment is invalid, leaving
              both the log and the fragments untouched.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRAGMENT_DIR = _REPO_ROOT / "changelog.d"
DEFAULT_CHANGELOG = _REPO_ROOT / "CHANGELOG.md"

#: Files in the fragment directory that are documentation, not fragments.
RESERVED_NAMES = frozenset({"README.md"})

FRAGMENT_NAME = re.compile(r"^(?P<number>\d+)-[a-z0-9]+(?:-[a-z0-9]+)*\.md$")
#: A fragment opens with the same level-3 entry heading the log already uses
#: (``### Fixed: ...``, ``### Added: ...``, ``### Docs: ...``). The wording is
#: not policed here: the log carries a dozen categories in practice, and this
#: script guards structure, not taxonomy.
FRAGMENT_HEADING = re.compile(r"^### \S")
UNRELEASED_HEADING = "## [Unreleased]"

#: Two blank lines separate sibling entries inside a section, matching the
#: existing hand-written layout of CHANGELOG.md.
ENTRY_SEPARATOR = "\n\n\n"


@dataclass(frozen=True)
class Fragment:
    """One accumulated changelog entry, read from ``changelog.d/``."""

    path: Path
    number: int
    body: str

    @property
    def name(self) -> str:
        return self.path.name


def collect_fragments(directory: Path = DEFAULT_FRAGMENT_DIR) -> list[Fragment]:
    """Read every fragment in ``directory``, newest number first.

    Ordering is by descending number then by name, so the assembled section
    reads newest-first like the rest of the log and is byte-identical across
    runs and machines (``Path.iterdir`` order is not guaranteed).

    Raises:
        ValueError: if the directory holds a file that is neither a reserved
            documentation file nor a validly named fragment.
    """
    if not directory.is_dir():
        return []

    fragments: list[Fragment] = []
    for path in sorted(directory.iterdir()):
        if path.name in RESERVED_NAMES or path.name.startswith("."):
            continue
        if not path.is_file():
            raise ValueError(f"{path}: fragment directory must hold files only")
        match = FRAGMENT_NAME.match(path.name)
        if match is None:
            raise ValueError(
                f"{path.name}: not a valid fragment name - expected "
                "'<number>-<slug>.md' (lowercase slug, words joined by '-')"
            )
        fragments.append(
            Fragment(
                path=path,
                number=int(match.group("number")),
                body=path.read_text(encoding="utf-8"),
            )
        )

    fragments.sort(key=lambda fragment: (-fragment.number, fragment.name))
    return fragments


def validate_fragment(fragment: Fragment) -> list[str]:
    """Return every contract violation in one fragment (empty list if valid)."""
    problems: list[str] = []
    lines = fragment.body.splitlines()
    stripped = [line for line in lines if line.strip()]

    if not stripped:
        problems.append(f"{fragment.name}: fragment is empty")
        return problems

    if not FRAGMENT_HEADING.match(stripped[0]):
        problems.append(
            f"{fragment.name}: first line must be a level-3 entry heading "
            f"('### <Category>: <summary>'), got {stripped[0]!r}"
        )

    for line in lines:
        if line.startswith("## "):
            problems.append(
                f"{fragment.name}: fragment may not contain a level-2 heading "
                f"({line!r}) - that would forge a version section"
            )
            break

    if sum(1 for line in lines if line.startswith("### ")) > 1:
        problems.append(
            f"{fragment.name}: fragment must hold exactly one '### ' entry - split multiple entries into one file each"
        )

    return problems


def validate_fragments(directory: Path = DEFAULT_FRAGMENT_DIR) -> list[str]:
    """Return every contract violation across the fragment directory."""
    try:
        fragments = collect_fragments(directory)
    except ValueError as exc:
        return [str(exc)]

    problems: list[str] = []
    for fragment in fragments:
        problems.extend(validate_fragment(fragment))
    return problems


def render(fragments: list[Fragment]) -> str:
    """Render fragments as the body of an ``[Unreleased]`` section."""
    return ENTRY_SEPARATOR.join(fragment.body.strip("\n") for fragment in fragments)


def insert_into_changelog(changelog_text: str, rendered: str) -> str:
    """Return ``changelog_text`` with ``rendered`` at the top of ``[Unreleased]``.

    Raises:
        ValueError: if the log has no ``## [Unreleased]`` heading to insert
            under, or more than one.
    """
    if not rendered:
        return changelog_text

    lines = changelog_text.splitlines()
    anchors = [i for i, line in enumerate(lines) if line.strip() == UNRELEASED_HEADING]
    if len(anchors) != 1:
        raise ValueError(f"CHANGELOG.md must hold exactly one '{UNRELEASED_HEADING}' heading, found {len(anchors)}")

    anchor = anchors[0]
    tail = lines[anchor + 1 :]
    # Drop the blank line(s) the heading is followed by; they are re-emitted
    # below so the seam has the same shape whether or not the section was empty.
    while tail and not tail[0].strip():
        tail.pop(0)

    head = "\n".join(lines[: anchor + 1])
    body = rendered.strip("\n")
    rest = "\n".join(tail).strip("\n")

    assembled = f"{head}\n\n{body}"
    if rest:
        assembled = f"{assembled}{ENTRY_SEPARATOR}{rest}"
    return assembled + "\n"


def apply(
    directory: Path = DEFAULT_FRAGMENT_DIR,
    changelog: Path = DEFAULT_CHANGELOG,
) -> list[Fragment]:
    """Fold fragments into ``changelog`` and delete the consumed files.

    Validation runs first and refuses the whole operation on any problem, so a
    malformed fragment can never leave the log half-written or a fragment
    deleted without its content landing.

    Returns:
        The fragments that were consumed (empty if there were none).

    Raises:
        ValueError: if any fragment is invalid or the log has no single
            ``[Unreleased]`` anchor. Nothing is written in that case.
    """
    problems = validate_fragments(directory)
    if problems:
        raise ValueError("refusing to assemble - invalid fragment(s):\n  " + "\n  ".join(problems))

    fragments = collect_fragments(directory)
    if not fragments:
        return []

    # Compute the new text before touching the filesystem: a bad anchor must
    # not cost a deleted fragment.
    updated = insert_into_changelog(changelog.read_text(encoding="utf-8"), render(fragments))

    changelog.write_text(updated, encoding="utf-8")
    for fragment in fragments:
        fragment.path.unlink()
    return fragments


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="validate fragments, write nothing")
    mode.add_argument("--print", dest="print_only", action="store_true", help="print the assembled section")
    mode.add_argument("--apply", action="store_true", help="fold fragments into CHANGELOG.md and delete them")
    parser.add_argument("--fragment-dir", type=Path, default=DEFAULT_FRAGMENT_DIR)
    parser.add_argument("--changelog", type=Path, default=DEFAULT_CHANGELOG)
    args = parser.parse_args(argv)

    problems = validate_fragments(args.fragment_dir)
    if problems:
        print("invalid changelog fragment(s):", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    fragments = collect_fragments(args.fragment_dir)

    if args.check:
        print(f"changelog fragments OK ({len(fragments)} pending)")
        return 0

    if args.print_only:
        print(render(fragments))
        return 0

    consumed = apply(args.fragment_dir, args.changelog)
    if not consumed:
        print("no changelog fragments to assemble")
        return 0
    print(f"assembled {len(consumed)} fragment(s) into {args.changelog.name}:")
    for fragment in consumed:
        print(f"  {fragment.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
