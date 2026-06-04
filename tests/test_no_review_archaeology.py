"""Repo hygiene: block review-archaeology from leaking into permanent artifacts.

Rationale
---------
Comments and test filenames live forever in the source tree; the GitHub PR
threads they reference do not. References like ``"review thread session.py:296"``
or ``test_pr224_review_blockers.py`` only make sense with the GitHub tab open,
and the line numbers / round labels go stale within a single mid-review push.

The contributor guide (AGENTS.md > Review Learnings) is explicit:

    review-round IDs (R1, R2, ...), reviewer names, and PR numbers belong in
    PRs/issues (ephemeral). Comments and test names must document INVARIANTS
    and BEHAVIOR (permanent).

A prose-only rule was widely violated (see issue tracker for the running
audit). A meta-test makes the rule enforceable: a rule with a test gets
followed.

What this checks
----------------
Two layers, both narrow on purpose:

1. **Filenames** under ``tests/`` matching ``test_pr<digits>*.py`` or
   ``*_review_r<digits>*.py`` or ``*_review_invariants_pr<digits>*.py``.
   Filenames should describe behavior (``test_acl_toctou.py``,
   ``test_validate_command.py``), not the PR number that introduced them.

2. **Source / test comments** containing the literal phrase ``review thread``
   or a ``@<reviewer-username>`` mention from the known-reviewers denylist.

What this does NOT check
------------------------
- Bare ``R1`` / ``R2`` / ``R7`` tokens in source (too many false positives:
  variables, register names, regression-suite labels). The high-signal
  ``review thread`` phrase covers the same archaeology in practice.
- ``PR #<n>`` references that genuinely document why a defensive check
  exists (``# See PR #168 for the bug this prevents``). Those are
  legitimate code archaeology, not review archaeology.
- Generic ``@<word>`` mentions: many are decorator references
  (``@ChoiceRegistry``, ``@tool``, ``@dataclass``) or technical terms.
  Only an explicit denylist of known reviewers is checked.

Allowlist
---------
A small allowlist below names files that currently violate one of the rules
and are tracked for cleanup elsewhere. The list should SHRINK over time -
new entries require a tracking issue. Do not add a file here without
referencing the issue that promises to remove it.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories to scan (source + tests; not docs, not third-party, not examples).
SCAN_DIRS = ("strands_robots", "tests", "tests_integ")


# ---------------------------------------------------------------------------
# Rule 1: test filenames must describe behavior, not PR/round
# ---------------------------------------------------------------------------

FILENAME_ARCHAEOLOGY = re.compile(
    r"""
    test_pr\d+               # test_pr224_*.py
    | _review_r\d+           # *_review_r1.py / *_review_r7.py
    | _review_invariants_pr\d+  # *_review_invariants_pr224.py
    """,
    re.VERBOSE,
)

# Files that currently violate Rule 1. Each entry MUST be tracked by an
# open issue that promises to rename / consolidate it. Do NOT add an entry
# here without filing the tracking issue first.
FILENAME_ALLOWLIST: frozenset[str] = frozenset(
    {
        # Tracked by review-archaeology cleanup issue (see PRs #225, #224
        # follow-ups). Targets:
        #   test_pr224_acl_toctou.py            -> test_acl_toctou.py
        #   test_pr224_review_blockers.py       -> merge into test_acl_config.py
        #   test_mesh_review_invariants_pr224.py -> merge into test_acl_config.py
        #   test_security_review_r1.py          -> test_security_dead_exports.py
        #   test_security_review_r7.py          -> test_security_defence_in_depth.py
        "tests/mesh/test_pr224_acl_toctou.py",
        "tests/mesh/test_pr224_review_blockers.py",
        "tests/mesh/test_mesh_review_invariants_pr224.py",
        "tests/mesh/test_security_review_r1.py",
        "tests/mesh/test_security_review_r7.py",
    }
)


# ---------------------------------------------------------------------------
# Rule 2: comments must not reference review threads or name reviewers
# ---------------------------------------------------------------------------

# Phrase ``review thread`` is the highest-signal review-archaeology marker
# in this repo's history. It is never a useful permanent comment.
REVIEW_THREAD_RE = re.compile(r"review[ _-]thread\b", re.IGNORECASE)

# Known reviewer GitHub usernames. Mentions of these in comments are review
# archaeology - the comment should describe WHY the code is the way it is,
# not WHO asked for it. Add usernames here as the reviewer set evolves.
KNOWN_REVIEWERS: frozenset[str] = frozenset(
    {
        "yinsong1986",
        # Add additional reviewer github logins here as needed.
    }
)

REVIEWER_MENTION_RE = re.compile(r"@(" + "|".join(re.escape(u) for u in sorted(KNOWN_REVIEWERS)) + r")\b")

# Files that currently violate Rule 2. Each entry MUST be tracked by an open
# issue. Cleanup is intentionally deferred for files that are mid-flight in
# active PR series (see scope-discipline note on the tracking issue).
COMMENT_ALLOWLIST: frozenset[str] = frozenset(
    {
        # Mid-flight in #195 mesh-split series; cleanup folded into the
        # relevant child PRs as they iterate.
        "strands_robots/mesh/_acl_config.py",
        "strands_robots/mesh/_zenoh_config.py",
        "strands_robots/mesh/core.py",
        "strands_robots/mesh/session.py",
        # Settled files - separate small comment-only PR will clean these.
        "strands_robots/simulation/models.py",
        "strands_robots/tools/download_assets.py",
        "strands_robots/policies/lerobot_local/resolution.py",
        # Test files that pin real regressions; comment cleanup folded into
        # test renames per the deferred-cleanup plan.
        "tests/mesh/test_acl_config.py",
        "tests/mesh/test_acl_shape_validation.py",
        "tests/mesh/test_default_acl_warning.py",
        "tests/mesh/test_pr224_review_blockers.py",
        "tests/mesh/test_robot_mesh_tool.py",
        "tests/tools/test_gr00t_inference.py",
        # This file itself documents the patterns it forbids.
        "tests/test_no_review_archaeology.py",
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iter_source_files() -> list[Path]:
    files: list[Path] = []
    for d in SCAN_DIRS:
        root = REPO_ROOT / d
        if not root.exists():
            continue
        for p in root.rglob("*.py"):
            if "__pycache__" in p.parts or ".venv" in p.parts:
                continue
            files.append(p)
    return files


def _is_comment_line(line: str) -> bool:
    """Heuristic: line is a ``#`` comment or inside a docstring/string literal.

    We accept some false positives (string literals that happen to contain
    ``review thread`` or ``@reviewer``) - in practice those are still
    archaeology and worth flagging.
    """
    stripped = line.lstrip()
    return (
        stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''")
        # Lines fully inside a triple-quoted block also count - we don't
        # parse python here, but most archaeology shows up either as ``#``
        # comments or inside module docstrings, both of which are caught
        # above for the line they start on. Conservative scan.
    )


# ---------------------------------------------------------------------------
# Rule 1 test
# ---------------------------------------------------------------------------


def test_no_review_archaeology_in_test_filenames() -> None:
    """Test filenames describe behavior, not the PR or review-round that bore them.

    A developer who breaks ACL TOCTOU should be able to find the regression
    test by searching for ``acl`` and ``toctou`` - not by knowing the PR
    number that introduced the fix. Names like ``test_pr224_acl_toctou.py``
    are review archaeology and should be renamed to
    ``test_acl_toctou.py`` (or merged into an existing acl test module).
    """
    offenders: list[str] = []
    for path in _iter_source_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in FILENAME_ALLOWLIST:
            continue
        if FILENAME_ARCHAEOLOGY.search(path.name):
            offenders.append(rel)

    if offenders:
        msg = [
            "Test filenames encode PR/review-round, not behavior:",
            *(f"  {o}" for o in sorted(offenders)),
            "",
            "Rename to describe what is being tested",
            "(e.g. test_acl_toctou.py, test_validate_command.py).",
            "If a deferred rename is required, add an entry to FILENAME_ALLOWLIST",
            "with a tracking issue.",
        ]
        raise AssertionError("\n".join(msg))


# ---------------------------------------------------------------------------
# Rule 2 test
# ---------------------------------------------------------------------------


def test_no_review_archaeology_in_comments() -> None:
    """Comments must not reference review threads or name reviewers.

    Comments document WHY code is shaped the way it is - the invariant or
    the bug being prevented. They do not document who suggested the change
    or which PR thread surfaced it; that information lives in the PR (and
    in ``git blame``).
    """
    offenders: list[tuple[str, int, str]] = []

    for path in _iter_source_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in COMMENT_ALLOWLIST:
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        for lineno, line in enumerate(text.splitlines(), start=1):
            if not _is_comment_line(line):
                continue
            if REVIEW_THREAD_RE.search(line):
                offenders.append((rel, lineno, line.strip()[:140]))
                continue
            if REVIEWER_MENTION_RE.search(line):
                offenders.append((rel, lineno, line.strip()[:140]))

    if offenders:
        msg = [
            "Review-archaeology in source comments:",
            *(f"  {rel}:{ln}: {snippet}" for rel, ln, snippet in offenders),
            "",
            "Rewrite the comment to describe the invariant or bug being",
            "prevented. Do NOT reference 'review thread X:NN' or",
            "'@<reviewer>' - that lives in the PR, not the source.",
        ]
        raise AssertionError("\n".join(msg))
