#!/usr/bin/env python3
"""Report every page under docs/ over the 900-word budget; exit 1 when any is.

Counts words the way ``wc -w`` does (whitespace-separated tokens) over the
page's markdown source, code fences included: the budget is about how much a
reader scrolls, and a fence scrolls too.

    python3 docs/hooks/word_budget.py [--limit 900] [--all]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1]
LIMIT = 900


def words(page: Path) -> int:
    """Whitespace-separated tokens in the page, the number ``wc -w`` prints."""
    return len(page.read_text(encoding="utf-8").split())


def main(argv: list[str] | None = None) -> int:
    """Print every page over the budget; return 1 when any is."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--limit", type=int, default=LIMIT, help=f"word budget per page (default {LIMIT})")
    parser.add_argument("--all", action="store_true", help="list every page, not only the ones over budget")
    args = parser.parse_args(argv)

    pages = sorted(p for p in DOCS.rglob("*.md") if "hooks" not in p.parts)
    counts = [(words(p), p.relative_to(DOCS)) for p in pages]
    over = [(n, p) for n, p in counts if n > args.limit]
    for n, p in sorted(counts if args.all else over, reverse=True):
        print(f"{n:6}  {p}" + ("  OVER" if n > args.limit else ""))
    print(f"\n{len(pages)} pages, {len(over)} over {args.limit} words")
    return 1 if over else 0


if __name__ == "__main__":
    sys.exit(main())
