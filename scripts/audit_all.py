#!/usr/bin/env python
"""Run every python audit in this directory and show what they FOUND.

The frontend has `npm run audit:all`; the python audits had no runner at all, so they only ever ran
when an agent remembered them by name - and in this repo the recurring failure is not a wrong rule, it
is a correct rule that never runs (three times in two days, which is why check-lib-wired.mjs exists).

Two lessons from `run-audits.mjs` are built in from the start:

  * A PASSING audit's news is printed. `audit_server_fields` exits 0 and its whole point is the list of
    dark fields it prints; discarding stdout because the word was "ok" is how the route audit's real
    finding sat unread for a day.
  * The count is reported as "X of Y ran", and a discovery that finds NOTHING to run is a failure, not
    a pass - the narrowed-and-empty rule this repo applies to every check that can be narrowed.

    .venv/bin/python scripts/audit_all.py            # non-zero only if an audit itself failed
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
#: Long enough for the slowest (audit_server_fields builds the app and enumerates hardware), short
#: enough that a hung audit cannot hold a loop iteration hostage.
TIMEOUT_S = 240.0
#: The words an audit uses to START a line that carries a finding, whatever its exit code was.
#: Matched as the FIRST TOKEN, not as a substring: a substring test forwarded every OpenCV
#: "OPENCV_AVFOUNDATION_SKIP_AUTH" line as news because it contains "SKIP", which buries the findings
#: under noise from a camera probe - the exact failure mode this runner exists to prevent.
NEWS_WORDS = frozenset({"NEWS", "FAIL", "SKIP", "note", "WARN", "warning:"})
#: …plus any "X of Y" line, because that is how this repo's audits disclose being NARROWED.
COUNT_RE = re.compile(r"\b\d+ of \d+\b")


def interesting(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return stripped.split(maxsplit=1)[0] in NEWS_WORDS or bool(COUNT_RE.search(stripped))


def main() -> int:
    audits = sorted(p for p in HERE.glob("audit_*.py") if p.name != Path(__file__).name)
    if not audits:
        print("  FAIL  no audit_*.py found — this is no information, not agreement")
        return 1

    ran = 0
    failed: list[str] = []
    for path in audits:
        name = path.stem.removeprefix("audit_")
        try:
            proc = subprocess.run(  # noqa: S603 - fixed interpreter, fixed directory
                [sys.executable, str(path)],
                capture_output=True, text=True, timeout=TIMEOUT_S, cwd=HERE.parent,
            )
        except subprocess.TimeoutExpired:
            print(f"  FAIL  {name}: still running after {TIMEOUT_S:.0f}s — killed, treat as unknown")
            failed.append(name)
            continue
        ran += 1
        verdict = "ok  " if proc.returncode == 0 else "FAIL"
        if proc.returncode != 0:
            failed.append(name)
        print(f"  {verdict}  {name}")
        # The news, from BOTH outcomes. stderr too: audit_collaborator_kwargs prints its own
        # narrowing warning there, above a "0 problems" summary that would otherwise read as green.
        for stream in (proc.stdout, proc.stderr):
            for line in stream.splitlines():
                if interesting(line):
                    print(f"        {line.strip()}")

    print(f"  {'FAIL' if failed else 'PASS'}  {ran} of {len(audits)} python audits ran"
          + (f" — failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
