#!/usr/bin/env python3
"""Repro for strands-labs/robots README 'What you get' links breaking on github.com.

The README 'What you get' table (README.md:91-104 at d94ef06) links each row to a
docs/*.md page via a relative path. On the published mkdocs site
(strands-labs.github.io/robots) those pages render fine. On github.com -- the
file viewer most first-time readers land on when they click a README link --
every one of those pages carries at least one mkdocs-only macro token
(`{{env_vars}}`, `{{extras:table}}`, `{{robot_cards}}`, `{{providers:table}}`,
`{{drawing:...}}`, `{{sim:first-robot-1|...}}`, ...). github.com renders
markdown literally, so the token stays verbatim and the user sees `{{env_vars}}`
instead of the 60+ env-var table the row promised.

The README names strands-labs.github.io/robots in its Documentation section
(README.md:107-110), AFTER the table whose links already fell through. A
first-time reader clicks a row link, lands on a page with no table, and has no
hint that there is a built docs site that would render it.

This script asserts the defect without a network call by reading the exact
.md files the README table points at and listing their un-expanded tokens.
A second step repeats the check against raw.githubusercontent.com (the bytes
github.com serves to its own markdown renderer) so the finding does not hide
behind a stale local checkout.

Exit code:
  0 - no README-linked docs page carries an un-expanded mkdocs token (defect fixed).
  1 - at least one README-linked docs page carries one (defect fires).
"""

from __future__ import annotations

import os
import re
import sys
import urllib.request
from pathlib import Path

# README-table rows and their target docs path, as they appear at d94ef06.
# Keep this list in sync with README.md:92-104 ("What you get").
ROWS: list[tuple[str, str]] = [
    ("Robots",         "docs/robots/index.md"),
    ("Policies",       "docs/learn/policies/index.md"),
    ("Teleoperation",  "docs/learn/hardware/teleoperation.md"),
    ("Recording",      "docs/learn/data/record.md"),
    ("Training",       "docs/learn/training/lerobot.md"),
    ("Simulation",     "docs/learn/simulation/index.md"),
    ("Mesh",           "docs/learn/mesh/fleet.md"),
    ("ROS 2",          "docs/learn/ros2.md"),
    ("Configuration",  "docs/reference/configuration.md"),
]

# Pages the "Documentation" paragraph (README.md:109) sends a new reader to.
DOC_ROWS: list[tuple[str, str]] = [
    ("Install (hero text)",     "docs/start/install.md"),
    ("Quickstart (first robot)","docs/start/first-robot.md"),
    ("Architecture",            "docs/concepts/architecture.md"),
]

TOKEN_RE = re.compile(r"\{\{\s*[^}\s][^}]*\}\}")
RAW_BASE = "https://raw.githubusercontent.com/strands-labs/robots/main/"


def tokens_in(text: str) -> list[str]:
    return sorted({m.group(0) for m in TOKEN_RE.finditer(text)})


def check_local(root: Path) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {}
    for label, rel in [*ROWS, *DOC_ROWS]:
        p = root / rel
        if not p.exists():
            hits[rel] = [f"<MISSING: {p}>"]
            continue
        toks = tokens_in(p.read_text(encoding="utf-8", errors="ignore"))
        if toks:
            hits[rel] = toks
    return hits


def check_github(network: bool = True) -> dict[str, list[str]]:
    """Fetch the same files from raw.githubusercontent.com and list their tokens.

    These are the bytes github.com itself hands to the markdown renderer, so the
    presence of a `{{...}}` token here is what the end user sees.
    """
    if not network:
        return {}
    hits: dict[str, list[str]] = {}
    for label, rel in [*ROWS, *DOC_ROWS]:
        url = RAW_BASE + rel
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                body = r.read().decode("utf-8", errors="ignore")
        except Exception as exc:  # pragma: no cover - network path
            hits[rel] = [f"<fetch error: {exc}>"]
            continue
        toks = tokens_in(body)
        if toks:
            hits[rel] = toks
    return hits


def main() -> int:
    root = Path(os.environ.get("STRANDS_ROBOTS_SRC", ".")).resolve()
    print(f"source dir : {root}")

    print()
    print("# Local check (what a `git clone` sees)")
    local = check_local(root)
    for rel, toks in local.items():
        # Trim long token lists for readability
        head = toks[:4]
        more = f" (+{len(toks)-4} more)" if len(toks) > 4 else ""
        print(f"  [{len(toks):3d} token(s)] {rel}: {', '.join(head)}{more}")

    print()
    print("# github.com check (via raw.githubusercontent.com)")
    want_net = os.environ.get("NO_NETWORK", "0") not in ("1", "true", "yes")
    remote = check_github(network=want_net)
    if not want_net:
        print("  (skipped: NO_NETWORK=1)")
    else:
        for rel, toks in remote.items():
            head = toks[:4]
            more = f" (+{len(toks)-4} more)" if len(toks) > 4 else ""
            print(f"  [{len(toks):3d} token(s)] {rel}: {', '.join(head)}{more}")

    broken = sorted(set(local) | set(remote))
    print()
    print(f"README 'What you get' table rows: {len(ROWS)}; broken on github.com: "
          f"{sum(1 for _, rel in ROWS if rel in local or rel in remote)}")
    print(f"'Documentation' intro pages: {len(DOC_ROWS)}; broken on github.com: "
          f"{sum(1 for _, rel in DOC_ROWS if rel in local or rel in remote)}")

    if broken:
        print()
        print(f"[FAIL] {len(broken)} README-linked docs page(s) carry un-expanded "
              f"mkdocs tokens that render literally on github.com.")
        return 1
    print()
    print("[OK] every README-linked docs page renders without un-expanded mkdocs "
          "tokens on github.com.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
