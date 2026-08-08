"""Measure the duplicate-claim intake check as invoked, in whichever tree runs it.

Each row is a real invocation against the live API. The first row is derived from
*this tree's* AGENTS.md, so the pair (guidance, script) is measured together rather
than the script alone.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

TREE = Path(__file__).resolve().parents[1]
SCRIPT = TREE / "scripts" / "check_duplicate_claim.py"
AGENTS = TREE / "AGENTS.md"
ISSUE = 2029
PR = 2028
AMBIENT = "huggingface/lerobot"


def documented_argv() -> list[str]:
    found = re.findall(r"python3 scripts/check_duplicate_claim\.py([^\n`]*)", AGENTS.read_text(encoding="utf-8"))
    assert len(found) == 1, found
    return [part.replace("<N>", str(ISSUE)) for part in found[0].split()]


def run(argv: list[str], ambient: str | None) -> dict[str, object]:
    env = dict(os.environ)
    env["GITHUB_TOKEN"] = os.environ["PAT_TOKEN"]
    if ambient is None:
        env.pop("GITHUB_REPOSITORY", None)
    else:
        env["GITHUB_REPOSITORY"] = ambient
    done = subprocess.run(
        [sys.executable, str(SCRIPT), *argv], cwd=TREE, env=env, capture_output=True, text=True, timeout=180
    )
    blob = done.stdout + done.stderr
    outcome = m.group(1) if (m := re.search(r"Outcome: \*\*([a-z-]+)\*\*", blob)) else None
    subject = m.group(1) if (m := re.search(r"\| (?:issue|pull request) \| (\S+) \|", blob)) else None
    compared = int(m.group(1)) if (m := re.search(r"\((\d+) compared\)", blob)) else None
    refusal = "must name the repository" in blob
    required = "--repo is required" in blob
    return {
        "argv": argv,
        "ambient": ambient,
        "exit": done.returncode,
        "outcome": outcome,
        "subject": subject,
        "compared": compared,
        "refused_inference": refusal,
        "required_flag": required,
        "blob": blob.strip(),
    }


rows = {
    "documented": run(documented_argv(), AMBIENT),
    "inferred_intake": run(["--issue", str(ISSUE)], AMBIENT),
    "explicit_intake": run(["--repo", "strands-labs/robots", "--issue", str(ISSUE)], AMBIENT),
    "review_mode": run(["--pr", str(PR)], "strands-labs/robots"),
    "nothing_to_infer": run(["--issue", str(ISSUE)], None),
}
payload = {"tree": str(TREE), "head": subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"], cwd=TREE, capture_output=True, text=True
).stdout.strip(), "documented_argv": documented_argv(), "rows": rows}
print("TREE:", TREE, "HEAD:", payload["head"])
out = Path(sys.argv[1])
out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(rows, indent=2))
