"""Repo hygiene: block emoji / orphan combining marks from package source.

History: AGENTS.md (PR #86 Review Learnings) codifies the rule:

    No emojis in user-facing strings - this is a project rule. Tool result
    dicts ({"content": [{"text": ...}]}), log messages, error messages: plain
    ASCII only. Agents read these strings programmatically; emojis just add
    tokenizer noise.

    Hunt orphan combining marks after any emoji sweep - U+23F1 (clock) plus
    U+FE0F (variation selector) renders as a stopwatch glyph; stripping the
    base U+23F1 leaves an invisible U+FE0F behind.

The rule was prose-only and widely violated. This test makes it executable.

What it scans
-------------
``strands_robots/`` (package source). Tests are intentionally NOT scanned -
test fixtures may legitimately need to assert on non-ASCII payloads.

What it flags
-------------
Codepoints in the Unicode "So" (Symbol, other) category - the standard emoji
class - plus U+FE0F (variation selector) regardless of category. CJK,
accented characters, and other letter-class non-ASCII would not be flagged
by this rule (they fall in L*, M*, N* categories), but in practice
``strands_robots/`` is English-only and none appear today.

Allowlist
---------
Files that are mid-flight in active review (the #195 mesh split, #196 gr00t
hardening, #209/#216 simulation cyclic-import work) are temporarily
allowlisted to avoid churning files under review. Each entry has a tracking
context. As those PRs land and follow-up sweeps clean a file, REMOVE it from
the allowlist - the test then enforces "no regression" for that file.

Goal: the allowlist shrinks to {} and stays there.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Only the package itself - test fixtures may assert on emoji payloads.
SCAN_DIRS = ("strands_robots",)

# Files temporarily allowed to contain emoji / orphan U+FE0F because they are
# under active review on an open PR. Sweep them in-place on those PRs (or in a
# follow-up surgical PR after they merge) and drop the entry. Do NOT add new
# entries here without an open tracking issue.
ALLOWED_FILES: frozenset[str] = frozenset(
    {
        # --- mesh files in flight on the #195 split (#221, #222, #225, #226,
        #     #227, #228) and on PR #304 (mesh tell sim dispatch). Sweep folds
        #     into those PRs to avoid merge churn.
        "strands_robots/tools/robot_mesh.py",
        # --- gr00t input validation in flight on PR #196.
        "strands_robots/tools/gr00t_inference.py",
        # --- simulation cyclic-import work in flight on PR #209 / #216.
        "strands_robots/simulation/base.py",
        "strands_robots/simulation/benchmark.py",
        "strands_robots/simulation/policy_runner.py",
        # --- not in flight; queued for the follow-up sweep PRs (simulation/
        #     first per issue #357, then tools/, then registry/policies).
        "strands_robots/simulation/__init__.py",
        "strands_robots/simulation/mujoco/simulation.py",
        "strands_robots/simulation/mujoco/physics.py",
        "strands_robots/simulation/mujoco/rendering.py",
        "strands_robots/simulation/mujoco/recording.py",
        "strands_robots/simulation/mujoco/randomization.py",
        "strands_robots/tools/lerobot_teleoperate.py",
        "strands_robots/tools/lerobot_camera.py",
        "strands_robots/tools/lerobot_calibrate.py",
        "strands_robots/tools/pose_tool.py",
        "strands_robots/tools/serial_tool.py",
        "strands_robots/policies/groot/policy.py",
        "strands_robots/registry/robots.py",
        "strands_robots/registry/user_registry.py",
        "strands_robots/benchmarks/libero/adapter.py",
    }
)


def _is_disallowed(ch: str) -> bool:
    """Return True if ``ch`` is an emoji-class symbol or an orphan VS-16.

    Implements the AGENTS.md-recommended check:

        unicodedata.category(ch).startswith("So") or ord(ch) == 0xFE0F

    "So" = Symbol, other (the emoji class). U+FE0F is the variation selector
    that turns a base codepoint into its emoji presentation; it is invisible
    on its own, which makes orphan FE0Fs the canonical PR #86 footgun.
    """
    if ord(ch) == 0xFE0F:
        return True
    return unicodedata.category(ch).startswith("So")


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


def test_no_emoji_in_package_source() -> None:
    """Fail if any package .py file contains an emoji or orphan U+FE0F.

    Plain ASCII only in tool-result strings, log messages, and error messages.
    If a non-ASCII codepoint is genuinely needed (e.g. a math symbol used in a
    docstring formula), document it and either narrow this check or move the
    text to a data file.
    """
    offenders: list[tuple[str, int, str, str]] = []

    for path in _iter_source_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in ALLOWED_FILES:
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        for lineno, line in enumerate(text.splitlines(), start=1):
            for col, ch in enumerate(line, start=1):
                if ord(ch) <= 0x7F:
                    continue
                if _is_disallowed(ch):
                    offenders.append((rel, lineno, f"U+{ord(ch):04X}", line.strip()[:120]))
                    break  # one finding per line is enough

    if offenders:
        msg = [
            "Emoji or orphan U+FE0F variation selector detected in package source.",
            "Project rule (AGENTS.md): plain ASCII in user-facing strings.",
            "If a file is mid-review on an open PR, add it to ALLOWED_FILES with a tracking note.",
            "",
        ]
        for rel, lineno, codepoint, snippet in offenders:
            msg.append(f"  {rel}:{lineno} [{codepoint}]: {snippet}")
        raise AssertionError("\n".join(msg))


def test_allowlist_entries_still_exist() -> None:
    """Drop allowlist entries when files are cleaned or deleted.

    Stale allowlist entries silently turn this test into a no-op for those
    paths. Whenever a sweep PR cleans a file, it must also remove the entry;
    whenever a file is deleted/renamed, the entry must follow. This guard
    fails fast on either drift.
    """
    missing = sorted(rel for rel in ALLOWED_FILES if not (REPO_ROOT / rel).is_file())
    if missing:
        raise AssertionError(
            "ALLOWED_FILES references paths that no longer exist - drop them:\n" + "\n".join(f"  {m}" for m in missing)
        )


def test_allowlist_entries_actually_violate() -> None:
    """Drop allowlist entries that are already clean.

    If a file is in ALLOWED_FILES but no longer contains any emoji/U+FE0F, the
    entry is dead weight - and worse, it suppresses regressions for that file.
    Force the entry to be removed once cleanup happens.
    """
    clean_but_listed: list[str] = []
    for rel in sorted(ALLOWED_FILES):
        path = REPO_ROOT / rel
        if not path.is_file():
            continue  # handled by the previous test
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if not any(_is_disallowed(ch) for ch in text):
            clean_but_listed.append(rel)

    if clean_but_listed:
        raise AssertionError(
            "ALLOWED_FILES contains files that are already emoji-clean - drop them:\n"
            + "\n".join(f"  {m}" for m in clean_but_listed)
        )
