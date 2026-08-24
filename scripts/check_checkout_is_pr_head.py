#!/usr/bin/env python3
"""Refuse work derived on a checkout the pull request's branch has moved past.

Why this exists
---------------
A pull request has *three* answers to "what is the head commit", not the two
``check_pr_head_is_current.py`` compares. That check reads the two the API
serves -- ``pullRequest { headRefOid }``, the value the pull request records,
against the tip of the branch in the head repository, the value that exists.
This one reads the third, which is the only one that decides what a fix gets
derived from: **the commit the local clone is actually on**.

The three disagree independently, and the third is the one with no signal:

===================================  ======================================
answer                               what moves it out of date
===================================  ======================================
``refs/pull/N/head``                 a mirror ref GitHub refreshes on its
                                     own schedule, so it trails a push to
                                     the fork branch
``pullRequest { headRefOid }``       normally reconciled within a second of
                                     a push; #2508 sat five hours behind
tip of the head repository's branch  nothing -- this is the value that
                                     exists, and what the other two are
                                     *for*
===================================  ======================================

Measured on #2678. A run checked the pull request out the obvious way and got
the mirror, while the API already served the commit that existed::

    git fetch origin pull/2678/head    ->  33f8bcf4     one commit behind
    pullRequest { headRefOid }         ->  0b070a05     the tip
    git merge-base --is-ancestor 33f8bcf4 0b070a05  ->  true, a pure lag

The cost is not a wasted fetch. The run believed it had checked out the branch,
grepped the symbol its review thread named, found it genuinely unused *on that
tree*, and derived a correct fix for source that no longer existed. What
surfaced the drift was ``git push`` being refused as non-fast-forward -- after
the work. A fix that happened to touch a different file would have pushed
cleanly.

And the answer it derived was worse than a duplicate. The thread was a CodeQL
"unused global ``_RATE_TAG``". Against the stale tree the fix is deletion, and
the suite agrees -- 18 passed, ruff clean. Against the tip the same finding
means the opposite: the constant was unused because the behavioural tests spelled
its value out at each call site, so nothing tied the joint under measurement to
the tag the parametrized sweep graded. Deletion would have passed CI while
removing the invariant the symbol existed to carry. An unused symbol in a test
file is often a *missing call site* rather than dead code, and the two fixes are
indistinguishable by test outcome.

#2520 records four instances of this class -- #2511 (one thread, four author
replies), #2566 (two runs deriving one fix), #2577 (two byte-identical commits,
one discarded), and #2678 above. In three of the four the only thing that
prevented a duplicate landing was a non-fast-forward push rejection, which is
load-bearing purely by accident: it fires because the mandated push shape is a
plain ``git push``, and both nearby shapes defeat it -- ``pull --rebase`` lands
an empty-diff commit on top, and ``--amend --force-with-lease`` deletes the
commit that already answered the thread.

Being ahead is not the finding
------------------------------
The comparison cannot be equality. A run that has committed its own work sits at
a commit the branch tip does not have yet, which is the ordinary state between a
commit and its push, and a check that reports it would cry wolf exactly when it
is being used correctly. So the question is ancestry, not identity: if the tip is
an ancestor of the checkout, the checkout contains everything the branch has and
the difference is the run's own unpushed work.

A tip that is *absent from the local object database* is not an ancestor and is
not unknowable either -- a local repository that has never fetched the tip
cannot contain it, so it is reported as stale rather than as indeterminate.

Outcomes
--------
``current``
    The checkout is the branch tip. The ordinary state at the start of a cycle,
    and the only one from which a thread's concern can be assessed.

``ahead``
    The tip is an ancestor of the checkout: the run's own commits, not yet
    pushed. Not a finding.

``stale-checkout``
    The finding. The branch has commits the checkout does not, so anything
    derived here answers a question about a tree somebody has already moved.

``unresolvable-head``
    The head repository or its branch could not be read -- a deleted fork, a
    deleted branch. Reported as its own outcome rather than folded into a pass,
    so a permissions or API-shape change cannot quietly turn this into a check
    that always agrees.

``unknown-checkout``
    The local ``HEAD`` could not be read; not a directory this can judge.

Why it reads the branch and not the mirror
------------------------------------------
Same reason ``check_pr_head_is_current.py`` reads the head repository's ref: the
value under suspicion cannot be its own witness. Here both of the cheap answers
are suspect -- the mirror is what produced the #2678 lag, and ``headRefOid`` is
what produced the #2508 one -- so the tip is resolved from the head repository's
own ref, through ``pullRequest { headRepository { nameWithOwner } headRefName }``.

Usage
-----
::

    python3 scripts/check_checkout_is_pr_head.py --repo strands-labs/robots --pr 2678
    python3 scripts/check_checkout_is_pr_head.py --repo strands-labs/robots --pr 2678 \\
        --checkout 33f8bcf4749a9146111294b4274be81326b50672

``--checkout`` defaults to ``git rev-parse HEAD`` in ``--git-dir`` (default: the
working directory). Exit 1 when the checkout is stale, so it can sit in front of
the work in a shell ``&&`` chain.

Unlike its sibling this takes no ``--all-open``: a checkout is a property of one
clone, so there is no population to sweep.

Pinned by tests/test_checkout_is_pr_head.py. See #2520.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass

CURRENT = "current"
AHEAD = "ahead"
STALE_CHECKOUT = "stale-checkout"
UNRESOLVABLE_HEAD = "unresolvable-head"
UNKNOWN_CHECKOUT = "unknown-checkout"

_API = "https://api.github.com/graphql"

# The remedy, named once so the report and the docstring cannot drift into
# describing different orderings for one rule.
WHAT_CLEARS_THIS: tuple[str, ...] = (
    "",
    "### What clears this",
    "",
    "Fetch the branch itself from the head repository, by name. Not",
    "`refs/pull/N/head` -- that mirror is what produced this finding on #2678,",
    "and re-fetching it can return the same stale commit again:",
    "",
    "```",
    "git fetch https://github.com/$HEAD_REPO.git $HEAD_REF",
    "git rev-parse FETCH_HEAD      # must equal headRefOid from the API",
    "```",
    "",
    "Then re-read the pull request's review threads against that commit before",
    "deriving anything. A thread is a statement about the commit it was written",
    "on; whether it is still true is a question about the tip. An unresolved",
    "thread whose concern the tip already answers is the normal steady state",
    "between a push and the next review pass, not a work item.",
    "",
    "Do **not** rebase or `--amend --force-with-lease` onto it. The commit the",
    "branch already carries may be the answer to the thread, and both of those",
    "shapes destroy it -- which is the failure #2520 records, where a plain",
    "`git push` being refused was the only thing that prevented it.",
)


@dataclass(frozen=True)
class Verdict:
    """The outcome and the two commits it was computed from."""

    outcome: str
    checkout: str | None
    tip: str | None

    @property
    def is_finding(self) -> bool:
        return self.outcome == STALE_CHECKOUT

    @property
    def summary(self) -> str:
        if self.outcome == CURRENT:
            return f"The checkout {_short(self.checkout)} is the branch tip. Derive from here."
        if self.outcome == AHEAD:
            return (
                f"The branch tip {_short(self.tip)} is an ancestor of the checkout "
                f"{_short(self.checkout)}, so the difference is this clone's own unpushed "
                "work. Not a finding."
            )
        if self.outcome == UNRESOLVABLE_HEAD:
            return (
                f"Could not read the head repository's branch, so the checkout "
                f"{_short(self.checkout)} could not be compared against anything. Not treated "
                "as a finding: a deleted fork or branch is a different problem."
            )
        if self.outcome == UNKNOWN_CHECKOUT:
            return (
                "Could not read the local HEAD, so there is no checkout to judge. Pass "
                "--checkout, or run this inside the clone."
            )
        return (
            f"The checkout is {_short(self.checkout)}, but the branch tip is "
            f"{_short(self.tip)} and the checkout does not contain it. Anything derived "
            "here answers a question about a tree that has already moved, and a review "
            "thread read against it may mean the opposite of what it means at the tip. "
            "Fetch the branch by name and re-read the threads before deriving a fix."
        )


def _short(oid: str | None) -> str:
    """Render a commit for a human without pretending an absent one is a commit."""
    if not oid:
        return "(unknown)"
    return f"`{oid[:8]}`"


def classify(checkout: str | None, tip: str | None, tip_is_ancestor: bool | None) -> Verdict:
    """Decide whether a checkout is a tree a fix may be derived on.

    ``tip_is_ancestor`` is passed in rather than computed so the decision stays a
    pure function of three readings; ``None`` means the local repository could
    not place the tip in its own history, which is not the same question as
    "could not read the tip" and is *not* indeterminate: a repository that does
    not contain a commit cannot have it as an ancestor, so it is stale.
    """
    if not checkout:
        return Verdict(UNKNOWN_CHECKOUT, checkout, tip)
    if not tip:
        return Verdict(UNRESOLVABLE_HEAD, checkout, tip)
    if checkout == tip:
        return Verdict(CURRENT, checkout, tip)
    if tip_is_ancestor:
        return Verdict(AHEAD, checkout, tip)
    return Verdict(STALE_CHECKOUT, checkout, tip)


def _graphql(query: str, variables: dict[str, object], token: str) -> dict:
    request = urllib.request.Request(
        _API,
        data=json.dumps({"query": query, "variables": variables}).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/vnd.github+json",
            "User-Agent": "strands-robots-check-checkout-is-pr-head",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - fixed API host
        payload = json.load(response)
    if payload.get("errors"):
        raise ValueError(f"GraphQL errors: {payload['errors']}")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ValueError("GraphQL response carried no data object")
    return data


_ONE_PR = """
query($owner:String!,$name:String!,$number:Int!){
  repository(owner:$owner,name:$name){
    pullRequest(number:$number){
      number headRefName headRefOid
      headRepository { nameWithOwner }
    }
  }
}
"""

_REF_TIP = """
query($owner:String!,$name:String!,$ref:String!){
  repository(owner:$owner,name:$name){
    ref(qualifiedName:$ref){ target { oid } }
  }
}
"""


def _split_repo(repo: str) -> tuple[str, str]:
    if repo.count("/") != 1 or not all(repo.split("/")):
        raise ValueError(f"--repo must be owner/name, got {repo!r}")
    owner, name = repo.split("/")
    return owner, name


def _git(args: Sequence[str], cwd: str) -> str | None:
    """Run git and return stdout, or ``None`` if it failed for any reason."""
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def local_head(cwd: str) -> str | None:
    """The commit the clone at ``cwd`` is on, or ``None`` if it is not a clone."""
    return _git(["rev-parse", "HEAD"], cwd)


def tip_is_ancestor_of_checkout(tip: str, checkout: str, cwd: str) -> bool | None:
    """Whether ``tip`` is in the history of ``checkout``, per the local repository.

    ``None`` when the local repository does not have the ``tip`` object at all,
    which the caller reads as stale rather than as unknown: a clone that has
    never fetched a commit cannot contain it. Distinguished from ``False`` --
    both are stale, but only ``False`` means the two commits genuinely diverged
    -- so the two causes stay legible in a report.
    """
    if _git(["cat-file", "-e", f"{tip}^{{commit}}"], cwd) is None:
        return None
    return _git(["merge-base", "--is-ancestor", tip, checkout], cwd) is not None


def branch_tip(head_repo: str, head_ref: str, token: str) -> str | None:
    """Return the tip of ``head_ref`` in ``head_repo``, or ``None`` if unreadable.

    Reads the head *repository's* ref. Neither ``refs/pull/N/head`` nor
    ``headRefOid`` will do: both are answers this check exists to catch out.
    """
    try:
        owner, name = _split_repo(head_repo)
    except ValueError:
        return None
    try:
        data = _graphql(
            _REF_TIP,
            {"owner": owner, "name": name, "ref": f"refs/heads/{head_ref}"},
            token,
        )
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError):
        return None
    repository = data.get("repository") or {}
    ref = repository.get("ref") or {}
    target = ref.get("target") or {}
    oid = target.get("oid")
    return oid if isinstance(oid, str) else None


def fetch_pull_request(repo: str, number: int, token: str) -> dict:
    owner, name = _split_repo(repo)
    data = _graphql(_ONE_PR, {"owner": owner, "name": name, "number": number}, token)
    repository = data.get("repository") or {}
    node = repository.get("pullRequest")
    if not isinstance(node, dict):
        raise ValueError(f"{repo}#{number} did not resolve to a pull request")
    return node


@dataclass(frozen=True)
class Report:
    """One evaluated checkout."""

    pr: int
    verdict: Verdict
    head_repo: str
    head_ref: str
    recorded: str | None

    def render(self, repo: str) -> str:
        lines = [
            "## Checkout against the pull request's branch tip",
            "",
            f"Outcome: **{self.verdict.outcome}**",
            "",
            self.verdict.summary,
            "",
            "| field | value |",
            "|---|---|",
            f"| pull request | {repo}#{self.pr} |",
            f"| head repository | `{self.head_repo}` |",
            f"| head branch | `{self.head_ref}` |",
            f"| local checkout | {_short(self.verdict.checkout)} |",
            f"| branch tip | {_short(self.verdict.tip)} |",
            f"| recorded headRefOid | {_short(self.recorded)} |",
        ]
        if self.recorded and self.verdict.tip and self.recorded != self.verdict.tip:
            lines += [
                "",
                f"Note: the pull request records {_short(self.recorded)} while its branch tip "
                f"is {_short(self.verdict.tip)}. That is the separate defect "
                "`check_pr_head_is_current.py` reports, whose remedy is a reopen and not a "
                "push; this check's verdict is against the tip either way.",
            ]
        if self.verdict.is_finding:
            lines += list(WHAT_CLEARS_THIS)
        return "\n".join(lines)


def evaluate(node: dict, token: str, checkout: str | None, cwd: str) -> Report:
    """Evaluate one pull request node against a checkout."""
    head_repository = node.get("headRepository") or {}
    head_repo = head_repository.get("nameWithOwner") or ""
    head_ref = node.get("headRefName") or ""
    recorded = node.get("headRefOid")
    tip = branch_tip(head_repo, head_ref, token) if head_repo and head_ref else None
    ancestry = tip_is_ancestor_of_checkout(tip, checkout, cwd) if tip and checkout and tip != checkout else None
    return Report(
        pr=int(node["number"]),
        verdict=classify(checkout, tip, ancestry),
        head_repo=head_repo or "(unknown)",
        head_ref=head_ref or "(unknown)",
        recorded=recorded,
    )


def _publish(report: str) -> None:
    print(report)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(report + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument("--pr", default=os.environ.get("PR_NUMBER", ""))
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN", ""))
    parser.add_argument(
        "--checkout",
        default="",
        help="Commit to judge. Defaults to git rev-parse HEAD in --git-dir.",
    )
    parser.add_argument(
        "--git-dir",
        default=".",
        help="Clone whose HEAD and object database are read (default: the working directory).",
    )
    args = parser.parse_args(argv)

    if not args.repo:
        print("check_checkout_is_pr_head: --repo is required", file=sys.stderr)
        return 0
    if not args.pr:
        print("check_checkout_is_pr_head: --pr is required", file=sys.stderr)
        return 0
    if not args.token:
        print("check_checkout_is_pr_head: no token; set --token or GITHUB_TOKEN", file=sys.stderr)
        return 0

    checkout = args.checkout or local_head(args.git_dir)
    try:
        node = fetch_pull_request(args.repo, int(args.pr), args.token)
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as exc:
        # Fails open: a check that cannot reach the API must not be the reason a
        # cycle stops, and its finding is advisory rather than a merge gate.
        print(f"check_checkout_is_pr_head: could not evaluate {args.repo}#{args.pr}: {exc}", file=sys.stderr)
        return 0
    report = evaluate(node, args.token, checkout, args.git_dir)
    _publish(report.render(args.repo))
    if report.verdict.is_finding:
        print(f"::warning title=Checkout is not the branch tip::{args.repo}#{report.pr}: {report.verdict.summary}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
