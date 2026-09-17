#!/usr/bin/env python3
"""Run the pull-request guards as one step of the one required check.

Why this exists
---------------
Every pull request here used to start twelve workflow runs: the required
``call-test-lint / Test and Lint`` and eleven advisory rows, each with its own
checkout, its own ``setup-python`` and one script that finishes in seconds.
Ten check rows per pull request, ten runner slots, and one of them
(``llm-input-safety``) ended with a step literally named ``Always pass``.

This script folds the guards into a single step that runs *before* the
install in ``test-lint.yml`` -- cheapest first, so a branch that breaks a
convention hears about it in under a minute instead of after a 30-minute
suite -- and keeps each guard's semantics and exit conditions exactly as its
workflow had them. The scripts themselves are unchanged; this file only
decides when to call them and folds their verdicts into one status.

Modes
-----
``guards``
    Pull-request events only. On any other event the guards have nothing to
    grade and say so. Runs, in cost order:

    * closing reference   ``scripts/check_closing_reference.py``   (API read)
    * LLM input safety    the two greps ``llm-input-safety.yml`` ran, now
                          failing on a finding instead of annotating one
    * changelog fragment  ``scripts/check_changelog_fragment.py``
    * merge-base overlap  ``scripts/check_merge_base_overlap.py``
    * lockfile parity     ``uv lock --check``

    Every guard runs; a finding in one does not hide a finding in another.
    Exit 1 when any guard reports a finding, 2 when any guard could not
    compute an answer (a missing tool, an unexpected status), else 0.

``api-drift``
    After the install. Runs griffe over the public Python surface and the
    AgentTool action-contract snapshot diff, and writes both reports to
    ``$GITHUB_STEP_SUMMARY``. Informational, as before: exit 0 unless the
    tooling itself broke. No pull-request comment is posted any more; the
    summary is the report.

Why the merge tree is now the right tree
----------------------------------------
The advisory workflows checked out the *base* branch to be sure the script
they ran existed (#1791: a branch forked before a gate landed does not carry
its script, and ``python3: can't open file`` renders as the same red X as a
finding). Inside the required check the tree under test is the pull
request's merge commit, which carries every script on the base tip by
construction, so the concern dissolves. The commits each guard *grades* are
still named explicitly -- ``--head`` is the pull request head, not the merge
commit, because the overlap and fragment checks are defeated by a merge
commit (``tests/test_merge_base_overlap.py::test_a_merge_commit_head_defeats_the_check``).

Standard library only.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO_ROOT / "scripts"

#: Exit statuses, shared with the scripts this folds: 0 clean, 1 finding,
#: 2 the guard could not compute an answer.
CLEAN, FINDING, BROKEN = 0, 1, 2

#: The two heuristics ``llm-input-safety.yml`` carried, verbatim. They are
#: regexes over agent-callable code, so they can false-positive; the answer to
#: a false positive is to restructure the call (bind the argument first), not
#: to make the guard advisory again -- a guard that cannot fail does not ship.
_SUBPROCESS_FSTRING = re.compile(r"subprocess\.(run|Popen|call|check_output|check_call)\([^)]*f[\"']")
_NAME_INTO_XML = re.compile(r"(\.format\([^)]*name|f[\"']<[a-z]+[^\"]*\{[a-z_]*name)")
_SUBPROCESS_DIRS = ("strands_robots/tools", "strands_robots/simulation", "strands_robots/mesh")
_XML_DIRS = ("strands_robots/simulation",)

#: Closes the griffe report. griffe compares two trees statically, so a flagged
#: removal may be the deliberate one this pull request is for; what it may not be
#: is silent.
_GRIFFE_FOOTER = (
    "> Static analysis; some flagged changes may be intentional. Confirm each, and add a migration "
    "note as a `changelog.d/` fragment or a deprecation notice before removal."
)


@dataclass(frozen=True)
class Verdict:
    guard: str
    status: int
    note: str = ""

    @property
    def label(self) -> str:
        return {CLEAN: "pass", FINDING: "FINDING", BROKEN: "BROKEN"}.get(self.status, f"exit {self.status}")


def _run(cmd: Sequence[str], *, env: dict[str, str] | None = None, cwd: Path | None = None) -> int:
    """Run a guard as a child process, streaming its output, and return its exit status."""
    print(f"$ {' '.join(cmd)}", flush=True)
    completed = subprocess.run(cmd, env=env, cwd=cwd or _REPO_ROOT, check=False)
    return completed.returncode


def _as_verdict(guard: str, status: int) -> Verdict:
    # The scripts reserve 0 and 1; anything else is the guard itself failing,
    # which is actionable by whoever owns the guard and not by the branch.
    if status in (CLEAN, FINDING):
        return Verdict(guard, status)
    return Verdict(guard, BROKEN, f"exited {status}")


# --------------------------------------------------------------------------- guards


def guard_closing_reference(*, repo: str, pr: str, token: str) -> Verdict:
    """A closing keyword that appears only in the title links nothing (#1961)."""
    if not (repo and pr and token):
        return Verdict("closing-reference", BROKEN, "needs GITHUB_REPOSITORY, PR_NUMBER and GITHUB_TOKEN")
    env = dict(os.environ, GITHUB_REPOSITORY=repo, PR_NUMBER=pr, GITHUB_TOKEN=token)
    # The title is deliberately not passed: the script reads it from the API so
    # a run that overlaps a title edit reports the post-edit verdict.
    env.pop("PR_TITLE", None)
    return _as_verdict(
        "closing-reference", _run([sys.executable, str(_SCRIPTS / "check_closing_reference.py")], env=env)
    )


def scan_llm_input(root: Path = _REPO_ROOT) -> list[tuple[str, int, str]]:
    """Return ``(path, line, kind)`` for every line the two heuristics match."""
    hits: list[tuple[str, int, str]] = []
    for dirs, pattern, kind in (
        (_SUBPROCESS_DIRS, _SUBPROCESS_FSTRING, "subprocess + f-string - LLM input must be validated first"),
        (_XML_DIRS, _NAME_INTO_XML, "name-like variable interpolated into XML/MJCF - validate ^[a-zA-Z0-9_-]+$ first"),
    ):
        for rel in dirs:
            for path in sorted((root / rel).rglob("*.py")):
                try:
                    lines = path.read_text(encoding="utf-8").splitlines()
                except UnicodeDecodeError:
                    continue
                for number, line in enumerate(lines, start=1):
                    if pattern.search(line):
                        hits.append((path.relative_to(root).as_posix(), number, kind))
    return hits


def guard_llm_input_safety() -> Verdict:
    """Unvalidated LLM input reaching a shell or an MJCF string (PR #85, #90)."""
    hits = scan_llm_input()
    for path, line, kind in hits:
        print(f"::error file={path},line={line}::{kind}")
    if hits:
        print(f"llm-input-safety: {len(hits)} finding(s); see AGENTS.md PR #85/#90 review learnings.")
        return Verdict("llm-input-safety", FINDING)
    print("llm-input-safety: no subprocess f-string or name-into-XML interpolation in agent-callable code.")
    return Verdict("llm-input-safety", CLEAN)


def guard_changelog_fragment(*, base_ref: str, head: str) -> Verdict:
    """An entry written straight into ``CHANGELOG.md``, or a fragment the assembler refuses."""
    cmd = [sys.executable, str(_SCRIPTS / "check_changelog_fragment.py"), "--base-ref", base_ref, "--head", head]
    return _as_verdict("changelog-fragment", _run(cmd))


def guard_merge_base_overlap(*, base_ref: str, head: str) -> Verdict:
    """A file the branch edits that its base also changed since they diverged (#1770)."""
    cmd = [sys.executable, str(_SCRIPTS / "check_merge_base_overlap.py"), "--base-ref", base_ref, "--head", head]
    return _as_verdict("merge-base-overlap", _run(cmd))


def guard_lockfile_parity() -> Verdict:
    """``uv.lock`` no longer resolves ``pyproject.toml`` (#2038)."""
    uv = shutil.which("uv")
    if uv is None:
        return Verdict("lockfile-parity", BROKEN, "uv is not on PATH; the workflow installs it before this step")
    return _as_verdict("lockfile-parity", _run([uv, "lock", "--check"]))


def run_guards(args: argparse.Namespace) -> int:
    if args.event != "pull_request":
        print(f"ci_guards: event is {args.event!r}, not a pull request; nothing to guard.")
        return CLEAN
    if not args.head:
        print("ci_guards: --head (the pull request head sha) is required on a pull_request event", file=sys.stderr)
        return BROKEN

    verdicts = [
        guard_closing_reference(repo=args.github_repo, pr=args.pr, token=args.token),
        guard_llm_input_safety(),
        guard_changelog_fragment(base_ref=args.base_ref, head=args.head),
        guard_merge_base_overlap(base_ref=args.base_ref, head=args.head),
        guard_lockfile_parity(),
    ]
    return _report(verdicts, title="Guards")


# --------------------------------------------------------------------------- api-drift


def _griffe_report(base_ref: str) -> tuple[str, int]:
    """Return ``(markdown, status)``; status 0 clean, 1 breakages, 2 griffe could not run."""
    try:
        import griffe  # type: ignore[import-not-found]
    except ImportError:
        return "## Breaking Change Analysis Skipped\n\ngriffe is not installed.", BROKEN
    try:
        old = griffe.load_git("strands_robots", ref=f"origin/{base_ref}", search_paths=["."])
        new = griffe.load("strands_robots", search_paths=["."])
    except Exception as exc:  # noqa: BLE001 - griffe's own errors are the report
        return f"## Breaking Change Analysis Skipped\n\ngriffe could not load both trees: `{exc}`", BROKEN
    breakages = list(griffe.find_breaking_changes(old, new))
    if not breakages:
        return "## No Breaking Changes Detected\n\nNo public API breaking changes found in this pull request.", CLEAN
    lines = ["## Breaking Change Warning", "", f"Found **{len(breakages)}** potential breaking change(s):", ""]
    lines += [b.explain() for b in breakages]
    lines += ["", "---", _GRIFFE_FOOTER]
    return "\n".join(lines), FINDING


def _agent_api_report(base_ref: str, workdir: Path) -> tuple[str, int]:
    """Snapshot the AgentTool action contract on base and head and diff them."""
    script = _REPO_ROOT / ".github" / "scripts" / "agent_api_snapshot.py"
    head_json, base_json = workdir / "api_head.json", workdir / "api_base.json"
    if _run([sys.executable, str(script), "snapshot", str(head_json)]) != 0:
        return "## AgentTool API Check\n\nThe head snapshot could not be built.", BROKEN
    base_tree = workdir / "base"
    if _run(["git", "worktree", "add", "--detach", str(base_tree), f"origin/{base_ref}"]) != 0:
        return "## AgentTool API Check Skipped\n\nThe base branch could not be checked out.", BROKEN
    try:
        # PYTHONPATH puts the base tree ahead of the editable install, so the
        # base snapshot reads the base *source*, not the head's package.
        env = dict(os.environ, PYTHONPATH=str(base_tree))
        if _run([sys.executable, str(script), "snapshot", str(base_json)], env=env, cwd=base_tree) != 0:
            return "## AgentTool API Check Skipped\n\nThe base snapshot could not be built.", BROKEN
        status = _run([sys.executable, str(script), "diff", str(base_json), str(head_json)])
    finally:
        _run(["git", "worktree", "remove", "--force", str(base_tree)])
    report = Path("/tmp/agent-api-changes.md")
    body = report.read_text(encoding="utf-8") if report.exists() else "## AgentTool API Check\n\nNo report produced."
    return body, (FINDING if status == 1 else CLEAN if status == 0 else BROKEN)


def run_api_drift(args: argparse.Namespace) -> int:
    if args.event != "pull_request":
        print(f"ci_guards: event is {args.event!r}, not a pull request; no base to diff against.")
        return CLEAN
    workdir = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "ci-guards"
    workdir.mkdir(parents=True, exist_ok=True)
    griffe_md, griffe_status = _griffe_report(args.base_ref)
    api_md, api_status = _agent_api_report(args.base_ref, workdir)
    for body in (griffe_md, api_md):
        print(body)
        _summary(body)
    for name, status in (("griffe", griffe_status), ("agent-api", api_status)):
        if status == FINDING:
            print(f"::warning title={name} drift::potential breaking change(s) - see the job summary")
    # Informational: the summary carries the signal. Only a broken tool fails.
    return BROKEN if BROKEN in (griffe_status, api_status) else CLEAN


# --------------------------------------------------------------------------- reporting


def _summary(text: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(text + "\n\n")


def _report(verdicts: list[Verdict], *, title: str) -> int:
    rows = [f"| {v.guard} | {v.label} | {v.note} |" for v in verdicts]
    table = "\n".join([f"## {title}", "", "| guard | verdict | note |", "|---|---|---|", *rows])
    print("\n" + table)
    _summary(table)
    worst = max((v.status for v in verdicts), default=CLEAN)
    if worst >= BROKEN:
        broken = ", ".join(v.guard for v in verdicts if v.status >= BROKEN)
        print(f"::error title=Guard broken::{broken} could not compute an answer; this is a defect in the guard.")
        return BROKEN
    if worst == FINDING:
        found = ", ".join(v.guard for v in verdicts if v.status == FINDING)
        print(f"::error title=Guards failed::{found}. Each remedy is self-clearing; see the log above.")
        return FINDING
    return CLEAN


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ci_guards.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("mode", choices=("guards", "api-drift"))
    parser.add_argument("--event", default=os.environ.get("GITHUB_EVENT_NAME", ""))
    parser.add_argument("--base-ref", default=os.environ.get("BASE_REF") or os.environ.get("GITHUB_BASE_REF") or "main")
    parser.add_argument(
        "--head", default=os.environ.get("HEAD_SHA", ""), help="pull request head sha, not the merge commit"
    )
    parser.add_argument("--pr", default=os.environ.get("PR_NUMBER", ""))
    parser.add_argument("--github-repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN", ""))
    args = parser.parse_args(argv)
    return run_guards(args) if args.mode == "guards" else run_api_drift(args)


if __name__ == "__main__":
    sys.exit(main())
