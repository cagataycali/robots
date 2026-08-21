#!/usr/bin/env python3
"""Did a commit quietly take assertions OUT of a test file?

WHY THIS EXISTS (Q106). On 2026-08-21 I wrote src/lib/passkey.test.mjs with `cat >` while claiming the
module was untested. It already had a test, written hours earlier by a parallel agent, and my redirect
deleted 134 lines and 31 assertions. Nothing complained: `npm test` was green before and after, because
A DELETED TEST IS A TEST THAT CANNOT FAIL, and the runner's file count even rose across the day because
other files were added. The lib-reachability guard proves every module is REACHED, never that a module's
assertions SURVIVE. The only witness was the commit log — so this reads the commit log.

Two agents editing one tree is the normal condition here, and the next one is as likely to do this as I
was. That is the whole justification: not that assertions are sacred, but that their removal is the one
regression class with no other witness.

NOT A GATE by default. The count is a regex over the word `assert`, so a legitimate refactor can lower it
— which is exactly what the second hit in the first run turned out to be (7f547a95 replaced two
`assert len(...) == N` with one shared `_assert_one_rollout(...)` helper that de-flaked the file: -1
assertion, strictly stronger). The DISCRIMINATOR that separates the two cases is whether the file also
SHRANK:

    lost assertions AND net line loss   -> SUSPECT: content was replaced, not refactored (my clobber:
                                          +88/-134)
    lost assertions while GROWING       -> likely an assertion extracted into a helper (+23/-2)

So the suspect class fails, the growing class is reported as a note, and the false-positive shape is named
here so no future loop spends an iteration "fixing" it.

Usage:
  python3 scripts/audit_test_assertion_loss.py                 # since 3 days ago
  python3 scripts/audit_test_assertion_loss.py --since '2026-08-20 00:00'
  python3 scripts/audit_test_assertion_loss.py --json
Exit: 0 clean (notes allowed) · 1 at least one SUSPECT · 2 nothing was scanned (see the narrowing law:
"0 problems" over 0 commits is not a pass).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

TEST_FILE = re.compile(r"(test_[\w-]*\.py|[\w.-]+\.test\.mjs|[\w-]+_test\.py)$")
ASSERT = re.compile(r"\bassert\b")


def classify(before: int, after: int, inserted: int, deleted: int, current: int | None = None) -> str:
    """'ok' | 'note' | 'suspect' | 'healed' — pure, so the rule is testable without a repo.

    A file that lost assertions while growing is almost always a helper extraction; one that lost them
    while shrinking had content REPLACED, which is the Q106 shape.

    `current` is the count at HEAD. History cannot be edited, so a repaired clobber would keep this tool
    red for ever (mine was restored in 548f2fff) and a permanently-red check is one people learn to
    ignore. If the file now carries at least as many assertions as before the loss, the wound is closed
    and the finding is reported as 'healed' — visible, not fatal.
    """
    if after >= before:
        return "ok"
    if current is not None and current >= before:
        return "healed"
    return "suspect" if deleted > inserted else "note"


#: audit_all.py forwards a finding line only if its FIRST TOKEN is one of its news words, so a verdict
#: whose tag is not in that vocabulary is invisible in the runner's summary — which is how 'healed'
#: findings vanished from the first real run of this audit inside audit_all (measured, not guessed).
#: tests/test_audit_test_assertion_loss.py imports audit_all's own NEWS_WORDS and checks these against it,
#: so if the runner's vocabulary ever changes, the divergence fails a test instead of silencing findings.
_TAGS = {"suspect": "FAIL   suspect", "healed": "note   healed", "note": "note   refactor?"}


def line_tag(verdict: str) -> str:
    """The prefix a finding is printed with — first token must be a word audit_all forwards."""
    return _TAGS[verdict]


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout


def _asserts(rev: str, path: str) -> int | None:
    r = subprocess.run(["git", "show", f"{rev}:{path}"], capture_output=True, text=True)
    return None if r.returncode else len(ASSERT.findall(r.stdout))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--since", default="3 days ago")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    # ONE numstat pass: probing every file of every commit is what made the first version time out.
    log = _git("log", "--since", args.since, "--reverse", "--format=@@%h", "--numstat")
    commit = None
    pairs: list[tuple[str, str, int, int]] = []
    commits: set[str] = set()
    for line in log.splitlines():
        if line.startswith("@@"):
            commit = line[2:].strip()
            commits.add(commit)
            continue
        parts = line.split("\t")
        if len(parts) == 3 and TEST_FILE.search(parts[2]) and parts[1].isdigit() and int(parts[1]) > 0:
            ins = int(parts[0]) if parts[0].isdigit() else 0
            pairs.append((commit or "", parts[2], ins, int(parts[1])))

    findings = []
    for rev, path, ins, dele in pairs:
        before, after = _asserts(f"{rev}^", path), _asserts(rev, path)
        if before is None or after is None:  # added or removed outright — not this audit's subject
            continue
        verdict = classify(before, after, ins, dele, _asserts('HEAD', path))
        if verdict != "ok":
            findings.append({"commit": rev, "file": path, "before": before, "after": after,
                             "inserted": ins, "deleted": dele, "verdict": verdict,
                             "subject": _git("log", "-1", "--format=%s", rev).strip()})

    suspects = [f for f in findings if f["verdict"] == "suspect"]
    if args.json:
        print(json.dumps({"commits": len(commits), "pairs_checked": len(pairs), "findings": findings}, indent=2))
    else:
        # Narrowing law: say what was actually examined, so a silent zero cannot pass for a pass.
        print(f"  {len(pairs)} (commit, test file) pair(s) with deletions checked across "
              f"{len(commits)} commit(s) since {args.since!r}")
        for f in findings:
            tag = line_tag(f["verdict"])
            print(f"  {tag} {f['commit']} {f['file'].split('/')[-1]}: {f['before']} -> {f['after']} "
                  f"assertions (+{f['inserted']}/-{f['deleted']} lines)")
            print(f"          {f['subject'][:100]}")
            if f["verdict"] == "healed":
                print("          (the assertions are back at HEAD — repaired, kept visible so the class stays known)")
            if f["verdict"] == "note":
                print("          (grew while losing an assertion — usually a helper extraction, see the docstring)")
        if not findings:
            print("  no test file lost assertions")
    if not commits:
        print(f"  nothing to check: no commits since {args.since!r}, which is not the same as clean")
        return 2
    return 1 if suspects else 0


if __name__ == "__main__":
    sys.exit(main())
