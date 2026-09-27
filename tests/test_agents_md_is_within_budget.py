"""AGENTS.md stays a contributor guide, not a log of every review.

Each rule added to AGENTS.md tends to arrive with a test pinning its wording, so
the file's size is also the size of a class of tests. A byte cap keeps the file
readable in one sitting and stops that class regrowing; rationale belongs in
the pull request that made a change.
"""

from __future__ import annotations

from pathlib import Path

AGENTS_MD = Path(__file__).resolve().parents[1] / "AGENTS.md"

#: Hard cap in bytes. Lower it when a cut lands; never raise it to fit a rule.
AGENTS_MD_MAX_BYTES = 30_000


def test_agents_md_is_within_budget() -> None:
    size = len(AGENTS_MD.read_bytes())
    assert size <= AGENTS_MD_MAX_BYTES, (
        f"AGENTS.md is {size:,} bytes, over the {AGENTS_MD_MAX_BYTES:,}-byte cap. Put the rationale "
        "in the pull request and keep only the rule here."
    )
