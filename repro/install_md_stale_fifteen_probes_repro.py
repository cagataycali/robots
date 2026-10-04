"""
repro: docs/start/install.md:70 and docs/start/index.md:20 say "fifteen probes"
while strands_robots.doctor.CHECKS has 17.

- #689 (closed) fixed docs/start/doctor.md (table + prose) and docs/reference/cli.md.
- Two sibling pages were missed. In the two weeks since, CHECKS grew from 16 to 17
  (iot_child_peers added), so the fossil is now TWO off, not one.

Run from the repo root:
    python repro/install_md_stale_fifteen_probes_repro.py
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# --- source of truth ---------------------------------------------------------
from strands_robots.doctor import CHECKS

truth = len(CHECKS)
cli_list = [name for name, _ in CHECKS]

# --- docs claims -------------------------------------------------------------
pages = {
    "docs/start/install.md": r"Fifteen probes",
    "docs/start/index.md": r"fifteen probes",
}

def extract(path: Path, needle: str) -> tuple[int, str] | None:
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if needle in line:
            return i, line.strip()
    return None

print(f"CHECKS source of truth: {truth} probes")
print("Order:", ", ".join(cli_list))
print()

missing = []
for rel, needle in pages.items():
    p = REPO / rel
    hit = extract(p, needle)
    if hit is None:
        print(f"[ok]  {rel}: no stale '{needle}' claim")
        continue
    lineno, text = hit
    print(f"[DEFECT] {rel}:{lineno}")
    print(f"         says: {text}")
    print(f"         actual CHECKS count: {truth}")
    print(f"         drift: {truth - 15} (off by two: 1 for #689-era, 1 for iot_child_peers)")
    missing.append((rel, lineno))

print()
# Prove via the CLI the user runs
print("--- strands-robots doctor --list (user-visible truth) ---")
import subprocess
r = subprocess.run(
    [sys.executable, "-m", "strands_robots", "doctor", "--list"],
    capture_output=True, text=True, timeout=30,
)
lines = [l for l in r.stdout.splitlines() if l.strip()]
print(f"printed {len(lines)} lines:")
for l in lines:
    print(f"  {l}")

if missing:
    print()
    print(f"DEFECT CONFIRMED: {len(missing)} docs page(s) carry a stale probe count.")
    sys.exit(1)

sys.exit(0)
