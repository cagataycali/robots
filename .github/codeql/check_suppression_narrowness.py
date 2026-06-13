#!/usr/bin/env python3
"""Assert the py/unsafe-cyclic-import suppression stays narrow.

Reads a SARIF file produced by running the CodeQL UnsafeCyclicImport query
with the config-level suppression dropped, and asserts that the set of files
firing the rule is EXACTLY the simulation triple.

If a new file starts participating in a static cycle, this exits non-zero with
a loud diagnostic. The maintainer must then either fix the new cycle properly
(preferred) or extend the documented suppression in .github/codeql/README.md.

Usage:
    python .github/codeql/check_suppression_narrowness.py cyclic-import.sarif
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RULE_ID = "py/unsafe-cyclic-import"

# The only files allowed to fire py/unsafe-cyclic-import. Keep in sync with
# .github/codeql/config.yml and .github/codeql/README.md.
EXPECTED_TRIPLE = {
    "strands_robots/simulation/base.py",
    "strands_robots/simulation/policy_runner.py",
    "strands_robots/simulation/benchmark.py",
}


def _extract_violating_files(sarif: dict) -> set[str]:
    """Return the set of source files that fired RULE_ID in the SARIF."""
    files: set[str] = set()
    for run in sarif.get("runs", []):
        for result in run.get("results", []):
            rule_id = result.get("ruleId", "")
            if RULE_ID not in rule_id:
                continue
            for loc in result.get("locations", []):
                uri = loc.get("physicalLocation", {}).get("artifactLocation", {}).get("uri", "")
                if uri:
                    files.add(uri.lstrip("./"))
    return files


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <sarif-file>", file=sys.stderr)
        return 2

    sarif_path = Path(argv[1])
    if not sarif_path.exists():
        print(f"error: SARIF file not found: {sarif_path}", file=sys.stderr)
        return 2

    sarif = json.loads(sarif_path.read_text())
    violating = _extract_violating_files(sarif)

    unexpected = violating - EXPECTED_TRIPLE
    missing = EXPECTED_TRIPLE - violating

    if unexpected:
        print(
            "FAIL: py/unsafe-cyclic-import fired on files OUTSIDE the documented "
            "simulation triple:\n  "
            + "\n  ".join(sorted(unexpected))
            + "\n\nThe config-level suppression in .github/codeql/config.yml is "
            "repository-wide, so these NEW cycles are being silently hidden in "
            "normal scans. Fix the new cycle (preferred) or extend the "
            "documented suppression in .github/codeql/README.md.",
            file=sys.stderr,
        )
        return 1

    if missing:
        # The triple no longer fires the rule. Not a regression, but the
        # suppression's scope changed and the docs should be updated.
        print(
            "WARNING: expected simulation-triple files no longer fire "
            "py/unsafe-cyclic-import:\n  "
            + "\n  ".join(sorted(missing))
            + "\n\nIf the static cycle was genuinely removed, drop the "
            "suppression from .github/codeql/config.yml and update the README.",
            file=sys.stderr,
        )
        # Soft signal only: do not fail CI on shrinkage.

    print(
        "OK: py/unsafe-cyclic-import suppression is narrow — violating file set "
        f"is the simulation triple ({len(violating)} files)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
