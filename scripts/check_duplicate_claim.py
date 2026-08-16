#!/usr/bin/env python3
"""Report whether an open pull request already closes a given issue.

Why this exists
---------------
``scripts/check_closing_reference.py`` reads one pull request in isolation: it
compares the issues a *title* claims against that same pull request's
``closingIssuesReferences``. Both of its inputs are per-pull-request, so a branch
that claims an issue **another open pull request already claims** passes every
check in this repository, and measured over the last 100 pull requests (#1867
through #2016) the notice has arrived after review every time.

Three pairs claimed one issue each -- six pull requests, three of them wasted::

    issue   pair            opened apart   abandoned   had reached
    #1942   #1944, #1946    8m 32s         #1944       APPROVED
    #1994   #1995, #1996    31m 58s        #1995       APPROVED
    #2007   #2015, #2016    4m 9s          #2016       APPROVED

The costly property is the last column. What a duplicate wastes is not the
authoring, it is a **review approval spent on a change that could never ship**,
and review is the scarcest resource here -- #1905 measures the same scarcity from
the other direction. Two of the three pairs could not both land regardless: the
closing comments on #1944 and #1995 each record a ``git merge-tree`` content
conflict, so the duplication was a merge failure waiting on whichever pull request
lost the race.

Every pair also opened inside one ~35-minute window, and that is what decides
where this belongs. The collision is an **intake** failure -- the second pull
request is opened before anything has observed the first -- not a drift that
accumulates over days. A check that reads two existing pull requests recovers the
review cost but not the authoring cost; a check run *before* the second one is
opened prevents both. So the primary entry point is ``--issue``, asked at intake,
and AGENTS.md step 1 is where it is asked.

Why the existing gate cannot see it
-----------------------------------
That check reads the title and the link set GitHub publishes, and that scope is
right for what it does. A duplicate claim is a property of the *set* of open pull
requests, so no single-pull-request assertion can reach it. The link set is still
the right thing to read -- it just has to be read across the open pull requests
rather than for one.

Why there is no rule about which of the two is at fault
------------------------------------------------------
Issue #2017 proposed failing "the newer of the two". Measured against the three
pairs, that names the wrong pull request twice::

    issue   older   newer   merged
    #1942   #1944   #1946   #1946   <- the newer one
    #1994   #1995   #1996   #1996   <- the newer one
    #2007   #2015   #2016   #2015

So in two of three the newer pull request is the one that survived, and an age
rule would have accused the eventual survivor. It is also not this check's
question: both numbers go in the report and which claim to drop stays with the
people who know what the two branches do.

That a repository-read token can see another pull request's link set is not
assumed. Issue #1961 records ``PullRequest.projectItems`` returning a false ``0``
under ``GITHUB_TOKEN``, so the lens was checked before relying on it: an Actions
token and a personal token return identical link sets for all the open pull
requests, including the one linking #1034. A false empty would turn this into a
silent no-op that always agrees, which is why an incomplete read is reported as
``unknown-claims`` rather than as a pass.

The open list is read through ``repository.pullRequests(states: OPEN)`` and not
through ``search``. Search is eventually consistent, so a pull request opened
seconds ago may not be indexed -- and the missing row would be a false clean in
exactly the ~35-minute window where every observed collision happened.

Two questions, one comparison
-----------------------------
``--issue N``
    Intake. *Before* authoring: does an open pull request already close #N?
    Nothing is excluded from the comparison, because no pull request for it exists
    yet.

``--pr N``
    Review. Do this pull request's own claims collide with another open one's?
    #N is excluded from its own comparison -- a pull request always shares its own
    claim.

What this reports, and what it deliberately does not
----------------------------------------------------
``no-claim``
    The pull request links no issue, so it can collide with nothing. Reachable
    only from ``--pr``; an issue is always its own claim.

``unique-claim``
    Nothing else claims it. The convention working.

``duplicate-claim``
    The finding.

``unknown-claims``
    A link set, or the open-pull-request list, could not be read completely. Not a
    finding -- an unreadable field is not evidence of a duplicate.

Out of scope, deliberately:

- **A pull request that claims nothing anywhere.** Two competing branches that
  both omit the keyword collide invisibly here. That is the same residual the
  sibling gate names and the one #1961 is about, and 18 of the last 30 merges
  would fail a rule requiring a claim, so this needs no answer to that question
  and cannot be dismissed by one.
- **Whether the issue exists, or is already closed.** A stale number is a
  different defect, and refusing it would report a finding against correct work
  whose issue someone else closed first.
- **Draft pull requests are included.** A draft's link is a real claim -- GitHub
  will close the issue when it merges -- so excluding drafts would hide a
  collision for as long as one side stays a draft.

Not wired to CI here, and that is a scope statement rather than an oversight. A
``pull_request`` job running ``--pr`` on ``opened``/``edited`` is the natural
follow-up and would cap the review cost of a collision this query did not
prevent; it needs a credential that can write ``.github/workflows/``, which is a
separate change. The intake half stands on its own: it is the half that stops the
work from being done twice, and it is complete as shipped.

Usage
-----
``--repo``    ``owner/name``. Required in intake mode; in ``--pr`` mode it
              defaults to ``$GITHUB_REPOSITORY``. See
              :func:`inferred_repository_refusal` for why the two modes differ.
``--issue``   an issue number, asked at intake. Mutually exclusive with ``--pr``.
``--pr``      a pull request number (default: ``$PR_NUMBER``).
``--token``   API token (default: ``$GITHUB_TOKEN``). Needs ``pull-requests: read``.

Exit status is ``1`` for ``duplicate-claim``, else ``0``. A usage error, including an
intake question whose repository was left to be inferred, exits ``2``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

API_ROOT = "https://api.github.com"

#: How many linked issues to ask for per pull request. A set longer than this is
#: treated as unreadable rather than as a short list, because a link set cut off
#: here could hide the very issue that collides and report a clean answer.
LINK_PAGE_SIZE = 100

#: How many open pull requests to read per page.
OPEN_PAGE_SIZE = 100

#: A bound on pagination, so a pathological repository cannot spin this job.
#: Reaching it is reported as ``unknown-claims`` -- an unread page is not
#: evidence of no collision.
MAX_OPEN_PAGES = 20

NO_CLAIM = "no-claim"
UNIQUE_CLAIM = "unique-claim"
DUPLICATE_CLAIM = "duplicate-claim"
UNKNOWN_CLAIMS = "unknown-claims"

#: ``closingIssuesReferences`` is GraphQL-only -- no REST field carries it.
_SELF_QUERY = """
query($owner: String!, $name: String!, $number: Int!, $links: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      number
      closingIssuesReferences(first: $links) {
        totalCount
        nodes { number }
      }
    }
  }
}
"""

_OPEN_QUERY = """
query($owner: String!, $name: String!, $links: Int!, $open: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(states: OPEN, first: $open, after: $after) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number
        closingIssuesReferences(first: $links) {
          totalCount
          nodes { number }
        }
      }
    }
  }
}
"""


class ClaimSetUnreadable(RuntimeError):
    """A link set, or the list of open pull requests, could not be read.

    Named apart from the builtin to keep the distinction explicit at the call
    site: this is "GitHub did not answer", not "the answer was empty". The two
    must not collapse, because an empty answer is a pass and an unanswered one is
    neither a pass nor a finding.
    """


@dataclass(frozen=True)
class Verdict:
    """The outcome and the claim sets it was computed from."""

    outcome: str
    #: The issues the pull request under test links.
    claimed: tuple[int, ...] = ()
    #: ``(issue, the other open pull requests linking it)``, sorted by issue.
    collisions: tuple[tuple[int, tuple[int, ...]], ...] = ()
    #: How many other open pull requests were compared against.
    scanned: int = 0
    detail: str = ""

    @property
    def is_finding(self) -> bool:
        return self.outcome == DUPLICATE_CLAIM

    @property
    def rivals(self) -> tuple[int, ...]:
        """Every other open pull request implicated, sorted and deduplicated."""
        return tuple(sorted({pr for _, prs in self.collisions for pr in prs}))

    @property
    def summary(self) -> str:
        if self.outcome == NO_CLAIM:
            return "This pull request links no issue, so it cannot duplicate another's claim."
        if self.outcome == UNIQUE_CLAIM:
            # Worded without a subject so it reads correctly for both entry points:
            # an issue checked at intake has no pull request to call "this" yet.
            return (
                f"{_issues(self.claimed)} "
                f"{'are' if len(self.claimed) > 1 else 'is'} claimed by no open pull request "
                f"({self.scanned} compared)."
            )
        if self.outcome == UNKNOWN_CLAIMS:
            return (
                f"Could not read every claim: {self.detail} Not treated as a finding, because "
                "an unreadable link set is not evidence that an issue is claimed twice."
            )
        # Semicolons rather than :func:`_join`, which would put an "and" between two
        # clauses that already each contain one ("... by #5 and #6 and #30 is ...").
        parts = "; ".join(f"{_issues((issue,))} is also claimed by {_pulls(prs)}" for issue, prs in self.collisions)
        return (
            f"{parts}. Two pull requests closing one issue cannot both "
            "land as written, and whichever loses the race spends a review approval on a "
            "change that will be closed."
        )


def _join(parts: Sequence[str]) -> str:
    """Render a clause list as ``a`` / ``a and b`` / ``a, b and c``."""
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _issues(numbers: Sequence[int]) -> str:
    """Render an issue list as ``#1`` / ``#1 and #2`` / ``#1, #2 and #3``."""
    return _join([f"#{n}" for n in numbers]) or "no issue"


def _pulls(numbers: Sequence[int]) -> str:
    """Render a pull-request list the same way, for the report's other half."""
    return _join([f"#{n}" for n in numbers]) or "no pull request"


def find_collisions(
    claimed: Sequence[int], others: Mapping[int, Sequence[int]]
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Return each claimed issue that another open pull request also claims.

    Deterministic in both axes -- issues ascending, and the pull requests
    claiming each ascending -- so the report reads the same on a re-run.
    """
    collisions = []
    for issue in sorted(set(claimed)):
        rivals = tuple(sorted(pr for pr, links in others.items() if issue in set(links)))
        if rivals:
            collisions.append((issue, rivals))
    return tuple(collisions)


def classify(
    claimed: Sequence[int] | None,
    others: Mapping[int, Sequence[int]] | None,
    detail: str = "",
) -> Verdict:
    """Decide which of the four states this pull request is in.

    Either argument being ``None`` means that half could not be read, which is
    its own outcome and folded into neither a pass nor a finding: a silent API or
    permission change must not be able to turn this check into a no-op that
    always agrees, nor into one that accuses every branch.
    """
    if claimed is None or others is None:
        return Verdict(UNKNOWN_CLAIMS, tuple(sorted(set(claimed or ()))), (), 0, detail)
    claimed_sorted = tuple(sorted(set(claimed)))
    if not claimed_sorted:
        return Verdict(NO_CLAIM, (), (), len(others))
    collisions = find_collisions(claimed_sorted, others)
    if not collisions:
        return Verdict(UNIQUE_CLAIM, claimed_sorted, (), len(others))
    return Verdict(DUPLICATE_CLAIM, claimed_sorted, collisions, len(others))


def _post(query: str, variables: dict[str, object], token: str) -> object:
    request = urllib.request.Request(
        f"{API_ROOT}/graphql",
        data=json.dumps({"query": query, "variables": variables}).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "strands-robots-check-duplicate-claim",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - fixed API host
        return json.load(response)


def _repository(payload: object) -> dict[str, object]:
    """Return the ``repository`` object, or refuse the answer."""
    if not isinstance(payload, dict):
        raise ClaimSetUnreadable("the API response was not a JSON object.")
    if payload.get("errors"):
        raise ClaimSetUnreadable(f"the API returned errors: {json.dumps(payload['errors'])[:400]}")
    repository = (payload.get("data") or {}).get("repository")
    if not isinstance(repository, dict):
        raise ClaimSetUnreadable("the response carried no repository.")
    return repository


def link_numbers(pull: Mapping[str, object]) -> tuple[int, ...]:
    """Return the issues one pull-request node links, refusing a truncated set.

    A set cut off at :data:`LINK_PAGE_SIZE` could omit the issue that collides,
    so it is refused rather than read short -- the one thing this check must never
    do is report clean because it did not look far enough.
    """
    references = pull.get("closingIssuesReferences") or {}
    if not isinstance(references, dict):
        raise ClaimSetUnreadable(f"#{pull.get('number')} carried no link set.")
    nodes = references.get("nodes") or []
    total = references.get("totalCount")
    if isinstance(total, int) and total > len(nodes):
        raise ClaimSetUnreadable(f"#{pull.get('number')}'s link set is truncated ({total} links, {len(nodes)} read).")
    return tuple(sorted({int(node["number"]) for node in nodes if isinstance(node, dict) and "number" in node}))


def resolve_claim(repo: str, pr: int, token: str) -> tuple[int, ...]:
    """Return the issues the pull request under test links.

    Read by number rather than from the open list, so the answer does not depend
    on the pull request having been indexed or on it fitting the first page.
    """
    owner, _, name = repo.partition("/")
    if not owner or not name:
        raise ClaimSetUnreadable(f"repository {repo!r} is not in owner/name form.")
    repository = _repository(
        _post(_SELF_QUERY, {"owner": owner, "name": name, "number": pr, "links": LINK_PAGE_SIZE}, token)
    )
    pull = repository.get("pullRequest")
    if not isinstance(pull, dict):
        raise ClaimSetUnreadable(f"no pull request {repo}#{pr} in the response.")
    return link_numbers(pull)


def resolve_open_claims(repo: str, token: str, pr: int | None = None) -> dict[int, tuple[int, ...]]:
    """Return ``{open pull request number: the issues it links}``, excluding ``pr``.

    ``pr`` of ``None`` excludes nothing, which is the intake question: *is anything
    already claiming this issue?* There is no pull request to leave out yet.

    Paginates rather than reading one page: a repository with more open pull
    requests than :data:`OPEN_PAGE_SIZE` would otherwise get a clean answer
    computed from a prefix of the set.
    """
    owner, _, name = repo.partition("/")
    if not owner or not name:
        raise ClaimSetUnreadable(f"repository {repo!r} is not in owner/name form.")
    claims: dict[int, tuple[int, ...]] = {}
    cursor: str | None = None
    for _ in range(MAX_OPEN_PAGES):
        repository = _repository(
            _post(
                _OPEN_QUERY,
                {
                    "owner": owner,
                    "name": name,
                    "links": LINK_PAGE_SIZE,
                    "open": OPEN_PAGE_SIZE,
                    "after": cursor,
                },
                token,
            )
        )
        page = repository.get("pullRequests")
        if not isinstance(page, dict):
            raise ClaimSetUnreadable("the response carried no open pull requests.")
        for node in page.get("nodes") or []:
            if not isinstance(node, dict) or "number" not in node:
                continue
            number = int(node["number"])
            if pr is not None and number == pr:
                continue
            claims[number] = link_numbers(node)
        info = page.get("pageInfo") or {}
        if not info.get("hasNextPage"):
            return claims
        cursor = info.get("endCursor")
        if not isinstance(cursor, str):
            raise ClaimSetUnreadable("the open pull request list is paged but carried no cursor.")
    raise ClaimSetUnreadable(f"more than {MAX_OPEN_PAGES * OPEN_PAGE_SIZE} open pull requests; the list was truncated.")


def render(verdict: Verdict, repo: str, pr: int | None = None, issue: int | None = None) -> str:
    """Render the report for this run.

    Exactly one of ``pr`` and ``issue`` names the subject: a pull request whose
    claims were compared, or -- at intake -- an issue checked before a pull
    request for it exists.
    """
    subject = f"| pull request | {repo}#{pr} |" if pr is not None else f"| issue | {repo}#{issue} |"
    lines = [
        "## Duplicate closing claim",
        "",
        f"Outcome: **{verdict.outcome}**",
        "",
        verdict.summary,
        "",
        "| field | value |",
        "|---|---|",
        subject,
        f"| issues it closes | {_issues(verdict.claimed)} |",
    ]
    # Only stated when a comparison happened. A pull request claiming nothing
    # short-circuits before the open list is read, and printing "0 compared"
    # there would read as "the check could not see the other pull requests".
    if verdict.outcome != NO_CLAIM:
        lines.append(f"| other open pull requests compared | {verdict.scanned} |")
    if verdict.is_finding and issue is not None:
        lines += [
            f"| already claimed by | {_pulls(verdict.rivals)} |",
            "",
            "### What this means",
            "",
            f"{_pulls(verdict.rivals)} is already open against {_issues((issue,))}. Read it before",
            "starting: if it does the work, there is nothing to author, and if it does not, say so",
            "there rather than opening a second pull request against the same issue. If a competing",
            "implementation is wanted on purpose, exactly one of the two should claim the close and",
            "the other should cross-reference instead (`per #N`, `towards #N`).",
        ]
    elif verdict.is_finding:
        lines += [
            f"| also claimed by | {_pulls(verdict.rivals)} |",
            "",
            "### What clears this",
            "",
            "1. Decide which pull request closes the issue. The other drops the keyword from",
            "   its description -- `per #N`, `follow-up to #N`, `towards #N` all still",
            "   cross-reference without linking. GitHub drops the link when the description is",
            "   saved and this check re-runs on `edited`, so either side can clear its own run",
            "   without a push.",
            "2. If one of the two is redundant, close it.",
            "",
            "Both pull requests are named above rather than one being blamed: measured over the",
            "last 100 pull requests, the newer of a duplicate pair is the one that merged in two",
            "of the three cases, so which to keep is not something this check can decide.",
        ]
    return "\n".join(lines)


def inferred_repository_refusal(inferred: str) -> str:
    """Return why an inferred repository cannot answer an intake question.

    ``$GITHUB_REPOSITORY`` names the repository a command is *running in*. For the
    ``--pr`` mode that is the right answer by construction -- a workflow reviewing a
    pull request runs in the repository the pull request lives in, which is why the
    default is kept there. Intake runs *before* any pull request exists, so it is a
    local invocation by whoever is about to do the work, and nothing ties their
    working directory to the repository the issue belongs to.

    The failure that follows is silent rather than loud, which is why this is a
    refusal rather than a warning. The check reads a different repository's open
    pull requests, finds none of them claiming that number, and reports
    ``unique-claim`` with exit ``0``. Nothing in that report distinguishes it from
    the answer to the question that was meant, and it only misleads in the
    reassuring direction: a spurious collision would be investigated and found
    nonexistent, whereas a missed one is invisible.

    Nor can the substitution be detected after the fact. An issue number alone does
    not name a repository, so there is no second source to compare the resolved one
    against, and issue numbers are dense enough that an unrelated repository very
    often has one at the same number -- which is also why confirming the issue
    *exists* would not be a reliable substitute for naming the repository, on top of
    reversing this script's deliberate decision not to read the issue at all.
    """
    return (
        "intake mode must name the repository: pass --repo owner/name. The environment "
        f"infers {inferred!r}, which is where this command is running rather than "
        "necessarily the repository the issue belongs to, and an intake check that reads "
        "the wrong repository reports no duplicate."
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # ``None`` rather than the environment value, so "was it supplied?" stays
    # answerable after parsing -- the intake refusal below turns on that, not on
    # what the repository resolves to.
    parser.add_argument("--repo", default=None)
    parser.add_argument("--pr", default=os.environ.get("PR_NUMBER", ""))
    parser.add_argument("--issue", default="")
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN", ""))
    args = parser.parse_args(argv)

    inferred = os.environ.get("GITHUB_REPOSITORY", "")
    repo = inferred if args.repo is None else args.repo

    if not repo:
        parser.error("--repo is required (or set GITHUB_REPOSITORY)")
    if bool(args.pr) == bool(args.issue):
        parser.error("pass exactly one of --pr (review a pull request) and --issue (intake)")
    if args.repo is None and args.issue:
        parser.error(inferred_repository_refusal(repo))
    if not args.token:
        parser.error("--token is required (or set GITHUB_TOKEN)")

    pr = int(args.pr) if args.pr else None
    issue = int(args.issue) if args.issue else None
    claimed: tuple[int, ...] | None
    others: dict[int, tuple[int, ...]] | None
    detail = ""
    try:
        # In intake mode the "claim" is the issue being considered, and nothing is
        # excluded from the comparison because no pull request for it exists yet.
        claimed = (issue,) if issue is not None else resolve_claim(repo, int(args.pr), args.token)
        others = resolve_open_claims(repo, args.token, pr) if claimed else {}
    except (ClaimSetUnreadable, urllib.error.URLError, urllib.error.HTTPError, ValueError, KeyError) as exc:
        claimed, others, detail = None, None, str(exc)
        print(f"check_duplicate_claim: {detail}", file=sys.stderr)

    verdict = classify(claimed, others, detail)
    report = render(verdict, repo, pr, issue)
    print(report)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(report + "\n")

    if verdict.is_finding:
        print(f"::error title=Two open pull requests claim one issue::{verdict.summary}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
