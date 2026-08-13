#!/usr/bin/env python3
"""Report the files a pull request edits that its base also changed since it branched.

Why this exists
---------------
``main`` went red at ``0e636f8`` from two pull requests that were each
individually green and textually non-conflicting. #1766 and #1763 both edit
``_recompile_preserving_state`` in
``strands_robots/simulation/mujoco/scene_ops.py``, for unrelated reasons. #1766
landed first. Every signal the merge gate offers then read green on #1763 --
``reviewDecision: APPROVED``, ``statusCheckRollup: SUCCESS``,
``mergeable: MERGEABLE``, ``mergeStateStatus: CLEAN`` -- and the squash still
broke the suite, because #1763 carried a *premise* test asserting the exact
defect #1766 had just fixed::

    FAILED test_a_tendon_driven_actuator_is_outside_the_joint_matched_id_scope
            AssertionError: assert 2 not in [2]

None of those four signals could have caught it. They are all computed against
the base the branch was tested on: #1763's checks ran against ``32dc3f5b``,
which predates #1766, so the first evaluation of the two changes *together* was
``main`` itself. ``mergeStateStatus: CLEAN`` in particular is a statement about
**text** -- git had no conflicting hunks to report, and it is not git's job to
know that one branch's assertion describes the other branch's bug.

What this computes
------------------
The overlap between two path sets, both taken from the branch's merge base ``M``
with its base branch:

- ``M..head`` -- what the pull request edits.
- ``M..base`` -- what landed on the base branch after the pull request branched.

A non-empty intersection does not prove the combination is broken. It proves
something weaker and still worth blocking on: **the combination has never been
compiled**, so every green check on the pull request is evidence about a tree
that is not the tree being merged. For the pair above the intersection is
exactly one entry, ``strands_robots/simulation/mujoco/scene_ops.py``, and one
``pytest`` invocation over the two touched test files would have caught it.

Why the remedy is cheap, and self-clearing
------------------------------------------
Merging the base branch into the pull request advances the merge base to the
base tip. The ``M..base`` set becomes empty, so the intersection does too, and
the checks then re-run against a base that *contains* the newly-landed commits.
The check needs no override to clear: doing the thing it asks for makes it pass.

This is the targeted form of branch protection's "Require branches to be up to
date before merging". That setting demands an update plus a full re-run before
*every* merge, which serialises merges and costs a ~14.7k-test suite each time.
This demands one only when the branch and its base actually edited the same
file, which is the only case where the base moving can invalidate a result.

Prose is reported but does not block
------------------------------------
An overlap confined to ``.md`` / ``.rst`` / ``.txt`` cannot change what the test
suite or the built package does; if two branches edit the same prose region, git
reports a conflict and the merge gate already stops it. Those paths are listed
in the report -- suppressing them entirely would hide a signal a reader may
want -- but they do not set the exit status, so a docs PR that happens to share
a file with a landed docs PR is not asked to re-run a full test suite for a
result that cannot change.

Why there is a sweep mode
-------------------------
Everything above is computed for **one** branch against its base. That leaves a
second topology with the same failure mode and no signal at all: two *open* pull
requests editing the same file. Neither has an ``M..base`` overlap, because
neither has landed, so this check reads green on both -- and the pair still has
the property the check exists to name. The combination has never been compiled,
and whichever merges first hands the other an overlap it will only be told about
after the fact.

Measured over the 19 open pull requests on 2026-08-13: 171 pairs, 7 sharing at
least one path, 3 of those prose-only, so **4** pairs a reader would want to
know about. Two were found by hand in earlier sessions at the cost of a
composition run each -- #2233 and #2235 both edit
``tests/simulation/isaac/test_motion_primitives.py``, where one carries a
``strict`` xfail for the defect the other fixes, so the two compose to
``[XPASS(strict)]`` and a red suite; #2224 and #2227 both edit
``strands_robots/simulation/safe_output.py``. Both were real, and nothing asked
for either.

``--all-open`` is that caller. It reads each open pull request's own path set,
intersects every pair, and reports the pairs sharing a behaviour-bearing path.
It needs no clone: the path set comes from the API rather than from git, so the
sweep runs from anywhere.

Why the path set comes from the API
----------------------------------
A pull request's file list *is* the three-dot diff, so it equals the ``M..head``
set this file already reasons about, with no fetch per branch. Verified against
git on ten open pull requests -- identical path sets on all ten.

``previous_filename`` is included, and that is load-bearing rather than
defensive. The API reports a rename as one entry naming only the **new** path,
while ``changed_paths`` above passes ``--no-renames`` precisely so a rename
contributes *both* names. Measured on #2057, which moved
``tests/simulation/test_args_docstring_completeness.py`` to ``tests/``: the API's
``filename`` values alone give 6 paths and miss the old name, and adding
``previous_filename`` gives 7, matching ``git diff --no-renames`` exactly. Taking
the new name only would silently drop the old one from the comparison -- the
missed-overlap direction, which is the one this file already refuses elsewhere.

An incomplete read is named, never reported as clean
----------------------------------------------------
The file list is paginated and GitHub caps it. A pull request whose paths were
only partly read could share a path with another and never show it, so a
truncated read is reported as **not evaluated** rather than folded into the
comparison. Same for a pull request whose lookup fails: named in the report and
excluded from the pairs, because a silent gap in coverage that reads as a clean
sweep is the failure mode this whole file is written against.

What a pair finding asks for, and why it does not block
------------------------------------------------------
Nothing, from either author. A shared path between two open pull requests is not
a defect in either one -- it is a fact about the order they merge in. Whichever
lands first is unaffected; the second then gets exactly the ``M..base`` finding
above, self-clearing in the usual way. So this reports for the same reason
``check_last_push_approval.py`` does: a gate whose remedy belongs to neither
branch would sit red on branches that have done nothing wrong. It is not in the
required set and should not be added to one. What it buys is that the reader
merging them knows *which* pairs want one composition run first.

Usage
-----
``--base-ref``  the branch being merged into (default ``main``). Resolved as
                ``origin/<ref>`` when that exists, else as ``<ref>``.
``--head``      the commit under test (default ``HEAD``). In CI this must be the
                pull request's *head* commit, never the
                ``refs/pull/<n>/merge`` commit ``actions/checkout`` produces by
                default -- that commit already contains the base tip, which
                drives the merge base to the base tip and the overlap to the
                empty set, so the check would pass unconditionally. CI *names*
                that commit rather than checking it out, and runs this script
                from the base branch instead: a branch that forked before a gate
                landed does not carry that gate's script (issue #1791). Sound
                because every input below is read from the object database and
                never from the working tree.
``--repo``      repository root (default: the current working directory).
``--all-open``  Sweep every open non-draft pull request instead of one branch,
                reporting pairs that edit the same file. Needs no clone. Cannot
                be combined with ``--head``: both name what to evaluate, so a
                conflicting pair is an error rather than one silently winning.
``--repo-slug`` ``owner/name`` for the sweep (default: ``$GITHUB_REPOSITORY``).
                Spelled differently from ``--repo`` above, which is a local
                filesystem path in this script -- reusing one name for both
                would make the sweep and the single-branch mode disagree about
                what ``--repo`` means.
``--token``     API token for the sweep (default: ``$GITHUB_TOKEN``). Needs
                ``pull-requests: read``.

Exit status is ``1`` when a behaviour-bearing path overlaps, else ``0``. In
``--all-open`` the same rule applies to the pairs: ``1`` when at least one pair
shares a behaviour-bearing path. Neither is in the required set.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

#: Suffixes whose overlap cannot change the outcome of the test suite or the
#: contents of the built package, and so is reported without blocking.
PROSE_SUFFIXES = frozenset({".md", ".rst", ".txt"})

#: REST host for the sweep. The single-branch mode reads only the object
#: database, so this is reached exclusively from ``--all-open``.
API_ROOT = "https://api.github.com"

#: Page cap for both paginated reads. A pull request whose file list is longer
#: than this is reported as not evaluated rather than compared on a partial set:
#: a path never read cannot be found to overlap, and a sweep that quietly
#: compares half a branch is indistinguishable from one that found nothing.
MAX_PAGES = 30


class GitError(RuntimeError):
    """A git invocation this script depends on did not succeed.

    Raised rather than returning a sentinel: every caller here needs the real
    commit or path set to say anything true, and a check that silently reports
    "no overlap" because it could not reach the base branch is worse than one
    that fails loudly. A missing base ref in CI is a workflow bug, not a
    property of the pull request.
    """


def _git(*args: str, repo: Path | None = None) -> str:
    """Run one git command and return its stdout, raising ``GitError`` on failure."""
    command = ["git"]
    if repo is not None:
        command += ["-C", str(repo)]
    command += list(args)
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise GitError(f"{' '.join(command)} exited {completed.returncode}: {completed.stderr.strip()}")
    return completed.stdout


def resolve_base_ref(base_ref: str, repo: Path | None = None) -> str:
    """Return the revision to treat as the base branch tip.

    Prefers the remote-tracking ref, because a CI checkout of a pull request
    head usually has no local branch for the base: ``actions/checkout`` fetches
    the base as ``refs/remotes/origin/<ref>`` and never creates ``<ref>``. Falls
    back to the bare name so the script is runnable in a normal local clone.
    """
    for candidate in (f"origin/{base_ref}", base_ref):
        try:
            _git("rev-parse", "--verify", "--quiet", f"{candidate}^{{commit}}", repo=repo)
        except GitError:
            continue
        return candidate
    raise GitError(f"cannot resolve base ref {base_ref!r} as either 'origin/{base_ref}' or '{base_ref}'")


def merge_base(base: str, head: str, repo: Path | None = None) -> str:
    """Return the commit where ``head`` diverged from ``base``."""
    revision = _git("merge-base", base, head, repo=repo).strip()
    if not revision:
        raise GitError(f"no merge base between {base!r} and {head!r} - is the history shallow?")
    return revision


def changed_paths(start: str, end: str, repo: Path | None = None) -> frozenset[str]:
    """Return the paths that differ between two commits.

    ``--no-renames`` is deliberate. With rename detection a file the base
    renamed appears only under its new name, so a pull request still editing the
    old name would not intersect it. Reporting a rename as its delete plus its
    add puts both names in the set, which is the conservative direction for a
    check whose failure mode is a missed overlap.
    """
    output = _git("diff", "--name-only", "--no-renames", f"{start}..{end}", repo=repo)
    return frozenset(line for line in output.splitlines() if line)


def is_prose(path: str) -> bool:
    """Whether a path is documentation, and so reported without blocking."""
    return Path(path).suffix.lower() in PROSE_SUFFIXES


def partition_overlap(paths: Iterable[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split overlapping paths into ``(behaviour_bearing, prose)``, each sorted.

    Sorting is what makes the report and the annotations reproducible across
    runs: ``git diff`` order follows the tree, and a set has no order at all.
    """
    ordered = sorted(set(paths))
    return (
        tuple(path for path in ordered if not is_prose(path)),
        tuple(path for path in ordered if is_prose(path)),
    )


def overlapping_paths(pr_paths: Iterable[str], base_paths: Iterable[str]) -> tuple[str, ...]:
    """Return the sorted paths edited both by the pull request and by its base."""
    return tuple(sorted(frozenset(pr_paths) & frozenset(base_paths)))


def render_report(
    *,
    base_ref: str,
    merge_base_sha: str,
    blocking: Sequence[str],
    prose: Sequence[str],
    base_change_count: int,
) -> str:
    """Render the Markdown report written to stdout and the CI job summary.

    Every multi-line paragraph below is built as a named local with explicit
    ``+`` rather than from adjacent literals inside the ``lines`` list. Implicit
    concatenation there is indistinguishable from a forgotten comma: a paragraph
    split across two elements silently becomes two report lines, and two
    paragraphs missing their separator silently become one. The join that
    produces the report cannot tell the difference, and neither can a reader.
    """
    lines = ["## Merge-base overlap check", ""]

    if not blocking and not prose:
        no_overlap = (
            f"No overlap. This branch edits nothing that `{base_ref}` has changed since "
            + f"the two diverged at `{merge_base_sha[:8]}` "
            + f"({base_change_count} path(s) changed on `{base_ref}` in that span)."
        )
        lines.append(no_overlap)
        lines.append("")
        lines.append("The checks on this branch were computed against a base that cannot have invalidated them.")
        return "\n".join(lines) + "\n"

    if blocking:
        heading = (
            f"This branch and `{base_ref}` have both changed **{len(blocking)}** "
            + f"behaviour-bearing path(s) since they diverged at `{merge_base_sha[:8]}`:"
        )
        why = (
            "Every check on this branch ran against a base that predates those commits, "
            + "so the combination has not been compiled. A green result here is evidence "
            + "about a different tree than the one that would be merged."
        )
        remedy = (
            "**To clear this:** merge "
            + f"`{base_ref}` into this branch and push. That advances the merge base, "
            + "re-runs the checks against a base containing the landed commits, and makes "
            + "this check pass. Run the tests covering the paths above first - that is "
            + "the cheap part, and it is what the check exists to prompt."
        )
        lines.append(heading)
        lines.append("")
        lines += [f"- `{path}`" for path in blocking]
        lines.append("")
        lines.append(why)
        lines.append("")
        lines.append(remedy)

    if prose:
        if blocking:
            lines.append("")
        prose_heading = (
            f"Also overlapping, not blocking ({len(prose)} documentation path(s)) - "
            + "prose cannot change what the suite or the package does, and a genuine "
            + "collision inside one would surface as a merge conflict:"
        )
        lines.append(prose_heading)
        lines.append("")
        lines += [f"- `{path}`" for path in prose]

    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class PairOverlap:
    """Two open pull requests and the paths they both edit, prose split out."""

    lower: int
    higher: int
    blocking: tuple[str, ...]
    prose: tuple[str, ...]

    @property
    def is_finding(self) -> bool:
        """Whether this pair shares a path that can change what the suite does."""
        return bool(self.blocking)


def _get(url: str, token: str) -> object:
    """Read one JSON document from the API."""
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "strands-robots-check-merge-base-overlap",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - fixed API host
        return json.load(response)


def resolve_open_pull_requests(repo_slug: str, token: str) -> list[int]:
    """Return the numbers of every open non-draft pull request, sorted.

    Drafts are excluded for the reason they are excluded from the last-push
    sweep: a draft cannot merge, so it cannot be the branch that lands first and
    hands the other an overlap. Sorted so the report is stable between runs and
    a diff of two reports shows changed verdicts rather than reordered rows.
    """
    numbers: list[int] = []
    for page in range(1, MAX_PAGES + 1):
        payload = _get(f"{API_ROOT}/repos/{repo_slug}/pulls?state=open&per_page=100&page={page}", token)
        rows = payload if isinstance(payload, list) else []
        for row in rows:
            if row.get("draft"):
                continue
            number = row.get("number")
            if isinstance(number, int):
                numbers.append(number)
        if len(rows) < 100:
            break
    return sorted(numbers)


def pull_request_paths(repo_slug: str, pull_request: int, token: str) -> tuple[frozenset[str], bool]:
    """Return ``(paths, complete)`` for one pull request's own edits.

    The set is what the pull request changes against its base, which is the same
    ``M..head`` set the single-branch mode computes from git. Both the new and
    the previous name of a renamed path are included, so the comparison matches
    the ``--no-renames`` semantics ``changed_paths`` chose for the same reason.

    ``complete`` is ``False`` when the file list was longer than this script
    reads. The caller must not compare a partial set, so it is returned rather
    than logged: the difference between "no shared paths" and "the shared path
    was on a page nobody read" is the whole reason to track it.
    """
    paths: set[str] = set()
    complete = False
    for page in range(1, MAX_PAGES + 1):
        payload = _get(f"{API_ROOT}/repos/{repo_slug}/pulls/{pull_request}/files?per_page=100&page={page}", token)
        rows = payload if isinstance(payload, list) else []
        for row in rows:
            name = row.get("filename")
            if isinstance(name, str) and name:
                paths.add(name)
            previous = row.get("previous_filename")
            if isinstance(previous, str) and previous:
                paths.add(previous)
        if len(rows) < 100:
            complete = True
            break
    return frozenset(paths), complete


def pairwise_overlaps(paths_by_pull_request: Mapping[int, Iterable[str]]) -> tuple[PairOverlap, ...]:
    """Return every pair of pull requests sharing a path, ordered by number.

    Pure over its mapping so the comparison is testable without the network, and
    it partitions through ``partition_overlap`` rather than re-deriving what
    counts as prose -- one rule, so the sweep and the single-branch mode cannot
    drift apart on the exemption that halves the report.
    """
    sets = {number: frozenset(paths) for number, paths in paths_by_pull_request.items()}
    found: list[PairOverlap] = []
    for lower, higher in itertools.combinations(sorted(sets), 2):
        shared = sets[lower] & sets[higher]
        if not shared:
            continue
        blocking, prose = partition_overlap(shared)
        found.append(PairOverlap(lower, higher, blocking, prose))
    return tuple(found)


def sweep(repo_slug: str, token: str) -> tuple[dict[int, frozenset[str]], list[int]]:
    """Read every open pull request's path set.

    Returns the sets it could read completely and the numbers it could not. A
    failure or a truncated read on one pull request is skipped rather than
    raised, because one rate-limited branch must not suppress a finding about
    the others -- and it is named by the caller rather than dropped.
    """
    paths_by_pull_request: dict[int, frozenset[str]] = {}
    unevaluated: list[int] = []
    for number in resolve_open_pull_requests(repo_slug, token):
        try:
            paths, complete = pull_request_paths(repo_slug, number, token)
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as error:
            print(f"check_merge_base_overlap: #{number} file list unreadable, not evaluated: {error}", file=sys.stderr)
            unevaluated.append(number)
            continue
        if not complete:
            print(
                f"check_merge_base_overlap: #{number} file list exceeded {MAX_PAGES} pages, not evaluated",
                file=sys.stderr,
            )
            unevaluated.append(number)
            continue
        paths_by_pull_request[number] = paths
    return paths_by_pull_request, unevaluated


def render_sweep(
    overlaps: Sequence[PairOverlap],
    evaluated: Sequence[int],
    unevaluated: Sequence[int],
    repo_slug: str,
) -> str:
    """Render the pairwise report, findings named up front.

    Paragraphs are built as named locals joined with explicit ``+`` for the same
    reason ``render_report`` does it: adjacent string literals inside a list of
    report lines are indistinguishable from a forgotten comma.
    """
    findings = [pair for pair in overlaps if pair.is_finding]
    quiet = [pair for pair in overlaps if not pair.is_finding]
    total_pairs = len(evaluated) * (len(evaluated) - 1) // 2

    lines = [
        "## Open pull request overlap sweep",
        "",
        f"Compared {total_pairs} pair(s) across {len(evaluated)} open non-draft " + f"pull request(s) in {repo_slug}.",
        "",
    ]

    if findings:
        named = ", ".join(f"#{pair.lower}+#{pair.higher}" for pair in findings)
        heading = f"**{len(findings)} pair(s) edit the same behaviour-bearing file:** {named}."
        why = (
            "Neither branch has an overlap with its base, so this check reads green on "
            + "both, and the pair still has the property it exists to name: the two have "
            + "never been compiled together. Whichever merges first is unaffected; the "
            + "second then gets the base overlap above, after the fact."
        )
        remedy = (
            "**This asks nothing of either author.** It is a merge-order note: before "
            + "merging both, run the tests covering the shared paths against the two "
            + "composed, which is the cheap part and the whole point."
        )
        lines += [heading, "", why, "", remedy, ""]
        lines += ["| pair | shared behaviour-bearing path(s) |", "|---|---|"]
        for pair in findings:
            shown = ", ".join(f"`{path}`" for path in pair.blocking)
            lines.append(f"| #{pair.lower} + #{pair.higher} | {shown} |")
    else:
        lines.append("No two open pull requests edit the same behaviour-bearing file.")

    if quiet:
        prose_heading = (
            f"Also overlapping, not counted ({len(quiet)} pair(s) sharing only "
            + "documentation) - prose cannot change what the suite or the package does, "
            + "and a genuine collision inside one would surface as a merge conflict:"
        )
        lines += ["", prose_heading, ""]
        for pair in quiet:
            shown = ", ".join(f"`{path}`" for path in pair.prose)
            lines.append(f"- #{pair.lower} + #{pair.higher}: {shown}")

    if unevaluated:
        named = ", ".join(f"#{number}" for number in unevaluated)
        lines += [
            "",
            f"Not evaluated ({len(unevaluated)}): {named}.",
            "A file list this run could not read completely is named rather than",
            "omitted: a path nobody read cannot be found to overlap, so dropping it",
            "silently would let a partial sweep read as a clean one.",
        ]

    return "\n".join(lines) + "\n"


def _emit(report: str) -> None:
    """Print a report, and append it to the CI job summary when there is one."""
    print(report, end="" if report.endswith("\n") else "\n")
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(report if report.endswith("\n") else report + "\n")


def _run_sweep(repo_slug: str, token: str) -> int:
    """Sweep the open pull requests for pairs editing one file."""
    if not repo_slug:
        print("::error::--all-open needs --repo-slug (or $GITHUB_REPOSITORY)", file=sys.stderr)
        return 1
    try:
        paths_by_pull_request, unevaluated = sweep(repo_slug, token)
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as error:
        # Listing the pull requests is the one read with no partial result to
        # report, so it fails loudly rather than reporting an empty sweep.
        print(f"::error::could not list open pull requests: {error}", file=sys.stderr)
        return 1

    overlaps = pairwise_overlaps(paths_by_pull_request)
    _emit(render_sweep(overlaps, sorted(paths_by_pull_request), unevaluated, repo_slug))

    findings = [pair for pair in overlaps if pair.is_finding]
    for pair in findings:
        shown = ", ".join(pair.blocking)
        print(
            f"::warning title=Two open pull requests edit one file::{repo_slug}#{pair.lower} "
            f"and #{pair.higher} both edit {shown}; the two have never been compiled together."
        )
    return 1 if findings else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Compute the overlap and return the process exit status."""
    parser = argparse.ArgumentParser(
        prog="check_merge_base_overlap.py",
        description="Report files a pull request edits that its base also changed since it branched.",
    )
    parser.add_argument("--base-ref", default="main", help="branch being merged into (default: main)")
    parser.add_argument("--head", default=None, help="commit under test (default: HEAD)")
    parser.add_argument("--repo", default=None, help="repository root (default: current directory)")
    parser.add_argument(
        "--all-open",
        action="store_true",
        help="sweep every open non-draft pull request for pairs editing one file",
    )
    parser.add_argument(
        "--repo-slug",
        default=os.environ.get("GITHUB_REPOSITORY", ""),
        help="owner/name for --all-open (default: $GITHUB_REPOSITORY)",
    )
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN", ""), help="API token for --all-open")
    args = parser.parse_args(argv)

    # Both name what to evaluate, so a conflicting pair is an error: silently
    # honouring one would report on a single branch when a sweep was asked for,
    # or the reverse, and neither is recoverable from the output.
    if args.all_open and args.head is not None:
        parser.error("--all-open and --head are mutually exclusive")

    if args.all_open:
        return _run_sweep(args.repo_slug, args.token)

    head = args.head if args.head is not None else "HEAD"
    repo = Path(args.repo) if args.repo is not None else None

    try:
        base = resolve_base_ref(args.base_ref, repo=repo)
        fork_point = merge_base(base, head, repo=repo)
        pr_paths = changed_paths(fork_point, head, repo=repo)
        base_paths = changed_paths(fork_point, base, repo=repo)
    except GitError as error:
        # Loud and non-zero: a check that cannot compute its answer must not
        # report the reassuring one.
        print(f"::error::merge-base overlap check could not run: {error}", file=sys.stderr)
        return 1

    blocking, prose = partition_overlap(overlapping_paths(pr_paths, base_paths))
    report = render_report(
        base_ref=args.base_ref,
        merge_base_sha=fork_point,
        blocking=blocking,
        prose=prose,
        base_change_count=len(base_paths),
    )

    _emit(report)

    for path in blocking:
        annotation = (
            f"::error file={path}::{path} was also changed on {args.base_ref} after this "
            + "branch diverged; the checks on this branch never compiled the two together."
        )
        print(annotation)

    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main())
