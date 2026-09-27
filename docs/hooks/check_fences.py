#!/usr/bin/env python3
"""Run every runnable ``python`` fence under docs/ and report.

A fence is runnable when its info string is exactly ``python`` (``python
title="sketch"`` and every other variant is skipped). Each fence runs in a fresh
interpreter, ``argv[1]`` or ``/Users/cagatay/robots/.venv/bin/python`` by
default, with ``PYTHONPATH`` set to the repository root so it imports this
checkout and ``MUJOCO_GL=cgl`` so MuJoCo renders headless on macOS. The timeout
per fence is 120 s.

Prints one row per fence (page, fence index, status, seconds) and exits 1 when
any fence fails or times out.

    python3 docs/hooks/check_fences.py [interpreter] [--only start/first-robot.md]
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"
DEFAULT_PYTHON = "/Users/cagatay/robots/.venv/bin/python"
TIMEOUT_S = 120
FENCE = re.compile(r"^```python[ \t]*\n(.*?)^```[ \t]*$", re.M | re.S)


def fences(page: Path) -> list[str]:
    """The body of every fence whose info string is exactly ``python``."""
    return [m.group(1) for m in FENCE.finditer(page.read_text(encoding="utf-8"))]


def run(code: str, python: str) -> tuple[str, float, str]:
    """(status, seconds, tail of output) for one fence."""
    env = {**os.environ, "PYTHONPATH": str(REPO), "MUJOCO_GL": os.environ.get("MUJOCO_GL", "cgl"), "PYTHONUNBUFFERED": "1"}
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
        handle.write(code)
        path = handle.name
    start = time.monotonic()
    try:
        proc = subprocess.run(
            [python, path], cwd=tempfile.gettempdir(), env=env, capture_output=True, text=True, timeout=TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        return "TIMEOUT", time.monotonic() - start, ""
    finally:
        Path(path).unlink(missing_ok=True)
    tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or [""]
    return ("PASS" if proc.returncode == 0 else "FAIL"), time.monotonic() - start, tail[0][:160]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("python", nargs="?", default=DEFAULT_PYTHON, help="interpreter to run fences with")
    parser.add_argument("--only", action="append", default=[], help="page path under docs/ (repeatable)")
    args = parser.parse_args(argv)

    pages = sorted(p for p in DOCS.rglob("*.md") if "hooks" not in p.parts)
    if args.only:
        wanted = {str((DOCS / o).resolve()) for o in args.only}
        pages = [p for p in pages if str(p.resolve()) in wanted]

    rows: list[tuple[str, int, str, float, str]] = []
    for page in pages:
        for index, code in enumerate(fences(page), start=1):
            status, seconds, tail = run(code, args.python)
            rows.append((str(page.relative_to(DOCS)), index, status, seconds, tail))
            print(f"{status:7} {seconds:6.1f}s  {page.relative_to(DOCS)} #{index}" + (f"  {tail}" if status != "PASS" else ""), flush=True)

    failed = [r for r in rows if r[2] != "PASS"]
    print()
    print(f"{len(rows)} fences on {len(pages)} pages: {len(rows) - len(failed)} pass, {len(failed)} fail")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
