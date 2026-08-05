#!/usr/bin/env python3
"""Refuse a pull request whose closing keyword only appears in its title.

Why this exists
---------------
GitHub parses closing keywords from a pull request's **body** and from its
**commit messages**. It never parses the title. A title reading
``... (closes #1891)`` therefore links nothing: the issue is referenced, the
timeline shows a cross-reference, and the pull request merges without closing
it. Nothing on either side says the claim was dropped, because a cross-reference
looks exactly like the beginning of a closing link.

Measured over the last 100 pull requests in this repository, 29 titles carry a
closing keyword followed by an issue number. 27 of them also linked the issue
and are unaffected. Two did not::

    #1894  "refactor(examples): remove the Isaac Replicator synth-data stub
            (closes #1891)"          closingIssuesReferences: []   #1891 still OPEN
    #1923  "fix(training/rl): the action head is sized and named from the
            action keys (closes #1912)"
                                     closingIssuesReferences: []   #1912 closed by hand

#1891 was still open two days after the pull request that says it closes it
merged. #1912 was closed separately, so the loss cost only the link. Both are the
same defect, and it is invisible from the pull request: the title reads as a
closing claim to every human who sees it, and the field that would contradict it
is one nobody opens.

The wider cost is that the merged record stops being countable. Issue #1961
measured the project board against the last 30 merges and found 18 with neither a
board item nor a closing link, so "what did this deliver" has no answer that does
not involve reading diffs. A dropped closing keyword is one of the two ways an
item gets into that set, and the only one a machine can catch.

Why it reads the link set and not the body
-----------------------------------------
The obvious implementation -- scan the body for the same keyword and compare --
**passes the incident it was written for**. #1894's body does contain the words
``closes #1891``, inside a code span:

    - [ ] #1891 stays open as the tracker (this PR uses `closes #1891` in the
      commit for traceability ...)

GitHub does not link from a code span, so the body scan and GitHub disagree on
exactly the pull request that matters, and the check would report clean while the
link was lost. Reimplementing enough of GitHub's markdown parser to get that
right -- fenced blocks, code spans, block quotes, HTML comments -- is a second
rulebook that can drift from the first, and it can only ever approximate an
answer GitHub already publishes.

So the body is not scanned at all. ``closingIssuesReferences`` **is** the link
set, and the only text this script parses is the title, where GitHub does nothing
and there is consequently no ground truth to read.

That the Actions token can read that field is not assumed. Issue #1961 records
``PullRequest.projectItems`` returning a false ``0`` under ``GITHUB_TOKEN``, so
the same lens was checked here before relying on it -- ``GITHUB_TOKEN`` and a
personal token return identical link sets for #1930 (two issues), #1960 (one) and
#1894 (none). A false zero would turn this check into an accusation against every
branch, which is why an ambiguous read is reported as ``unknown-links`` rather
than as a finding.

What this reports, and what it deliberately does not
---------------------------------------------------
``no-claim``
    The title carries no closing keyword before an issue number. 71 of the last
    100 pull requests. Nothing to say.

``linked``
    Every issue the title claims to close is in the link set. 27 of the last 100.
    The convention working.

``title-only-claim``
    The finding: the title claims to close an issue that the pull request does
    not link. Two of the last 100.

``unknown-links``
    The link set could not be read, or was truncated by the page size. Not a
    finding -- an unreadable field is not evidence of a dropped claim.

Three things are deliberately out of scope:

- **A pull request that claims nothing anywhere.** Whether every change must
  trace to an issue is an open question (#1961, decision B) and 18 of the last 30
  merges would fail such a rule today. This check reads only what a title already
  claims, so it needs no answer to that question and cannot be dismissed by one.
  #1885 -- merged with no closing keyword at all while #1871 stayed open -- is in
  that class and stays uncaught here. Naming it is the honest scope statement:
  this closes one of the two ways a link is lost.
- **Cross-repository (``owner/repo#12``) and URL reference forms.** Every
  observed instance used a bare ``#N``, and a cross-repository claim would have
  to resolve another repository's numbering to be comparable.
- **Whether the claimed issue exists, or is already closed.** A stale number in a
  title is a different defect, and refusing it would make this check fail on a
  correct pull request whose issue someone else closed first.

Unlike the last-push-approval report, this is **self-clearing**: moving the
keyword into the body makes GitHub create the link, and the workflow re-runs on
``edited``, so the branch author alone can turn it green. That is why it exits
non-zero and annotates as an error rather than a warning.

Usage
-----
``--repo``   ``owner/name`` (default: ``$GITHUB_REPOSITORY``).
``--pr``     pull request number (default: ``$PR_NUMBER``).
``--title``  the title to read (default: ``$PR_TITLE``; falls back to the pull
             request's own title). Passed through the environment rather than
             interpolated into the workflow's shell, because it is author-
             controlled text.
``--token``  API token (default: ``$GITHUB_TOKEN``). Needs ``pull-requests:
             read``.

Exit status is ``1`` for ``title-only-claim``, else ``0``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass

API_ROOT = "https://api.github.com"

#: The keywords GitHub acts on, spelled out rather than built from stems so the
#: set is greppable against GitHub's own documented list.
CLOSING_KEYWORDS = (
    "close",
    "closes",
    "closed",
    "fix",
    "fixes",
    "fixed",
    "resolve",
    "resolves",
    "resolved",
)

#: A closing claim: one keyword immediately followed by an issue number, with at
#: most a colon and whitespace between them. Adjacency is what GitHub requires
#: and it is what keeps this off ordinary titles -- "fix(sim): ... (#1948)" is
#: not a claim because "(sim)" sits between the keyword and the number, and
#: "follow-up to #1722" has no keyword at all. Both forms are common here.
#: The leading word boundary keeps "prefixes #12" and "unfixed #12" out.
_CLAIM = re.compile(
    r"\b(?:clos(?:e|es|ed)|fix(?:es|ed)?|resolv(?:e|es|ed))\b\s*:?\s*#(\d+)\b",
    re.IGNORECASE,
)

#: How many linked issues to ask for. Above this the answer is treated as
#: unreadable rather than as a short list, because a truncated link set would
#: manufacture a finding out of a claim that is in fact linked.
LINK_PAGE_SIZE = 100

NO_CLAIM = "no-claim"
LINKED = "linked"
TITLE_ONLY_CLAIM = "title-only-claim"
UNKNOWN_LINKS = "unknown-links"

_QUERY = """
query($owner: String!, $name: String!, $number: Int!, $first: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      title
      closingIssuesReferences(first: $first) {
        totalCount
        nodes { number }
      }
    }
  }
}
"""


class LinkSetUnreadable(RuntimeError):
    """The link set could not be read.

    Named apart from the builtin to keep the distinction explicit at the call
    site: this is "GitHub did not answer", not "the answer was empty". The two
    must not collapse, because an empty answer is a finding and an unanswered
    one is not.
    """


@dataclass(frozen=True)
class Verdict:
    """The outcome and the two number sets it was computed from."""

    outcome: str
    claimed: tuple[int, ...] = ()
    linked: tuple[int, ...] = ()
    unlinked: tuple[int, ...] = ()
    detail: str = ""

    @property
    def is_finding(self) -> bool:
        return self.outcome == TITLE_ONLY_CLAIM

    @property
    def summary(self) -> str:
        if self.outcome == NO_CLAIM:
            return "The title claims to close no issue, so there is no claim to lose."
        if self.outcome == LINKED:
            return (
                f"The title claims to close {_issues(self.claimed)}, and the pull request links "
                f"{'them' if len(self.claimed) > 1 else 'it'}. Merging will close "
                f"{'them' if len(self.claimed) > 1 else 'it'}."
            )
        if self.outcome == UNKNOWN_LINKS:
            return (
                f"Could not read which issues this pull request links: {self.detail} "
                "Not treated as a finding, because an unreadable link set is not evidence "
                "that a claim was dropped."
            )
        return (
            f"The title says this pull request closes {_issues(self.unlinked)}, but "
            f"{'those issues are' if len(self.unlinked) > 1 else 'that issue is'} not in its "
            "closing-issue links. GitHub reads closing keywords from the body and from commit "
            f"messages, never from the title, so merging will leave "
            f"{'them' if len(self.unlinked) > 1 else 'it'} open."
        )


def _issues(numbers: Sequence[int]) -> str:
    """Render a number list as ``#1`` / ``#1 and #2`` / ``#1, #2 and #3``."""
    rendered = [f"#{n}" for n in numbers]
    if not rendered:
        return "no issue"
    if len(rendered) == 1:
        return rendered[0]
    return ", ".join(rendered[:-1]) + " and " + rendered[-1]


def title_claims(title: str) -> tuple[int, ...]:
    """Return the issue numbers the title claims to close, sorted and deduplicated.

    One keyword governs one number: ``closes #1506, closes #1516`` claims both
    (that is #1930's title, and both were linked), while ``fixes #12 and #13``
    claims only #12 -- which is also how GitHub reads a body, so a title written
    that way was already only ever going to close one of them.
    """
    return tuple(sorted({int(match.group(1)) for match in _CLAIM.finditer(title)}))


def unlinked_claims(claimed: Sequence[int], linked: Sequence[int]) -> tuple[int, ...]:
    """Return the claimed numbers absent from the link set, sorted."""
    return tuple(sorted(set(claimed) - set(linked)))


def classify(title: str, linked: Sequence[int] | None, detail: str = "") -> Verdict:
    """Decide which of the four states this pull request is in.

    ``linked`` of ``None`` means the link set could not be read, which is its own
    outcome and not folded into either a pass or a finding: a silent API or
    permission change must not be able to turn this check into a no-op that
    always agrees, nor into one that accuses every branch.
    """
    claimed = title_claims(title)
    if not claimed:
        return Verdict(NO_CLAIM)
    if linked is None:
        return Verdict(UNKNOWN_LINKS, claimed, (), (), detail)
    linked_sorted = tuple(sorted(set(linked)))
    unlinked = unlinked_claims(claimed, linked_sorted)
    if not unlinked:
        return Verdict(LINKED, claimed, linked_sorted)
    return Verdict(TITLE_ONLY_CLAIM, claimed, linked_sorted, unlinked)


def _post(url: str, payload: dict[str, object], token: str) -> object:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "strands-robots-check-closing-reference",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - fixed API host
        return json.load(response)


def resolve_pull_request(repo: str, pr: int, token: str) -> tuple[str, tuple[int, ...]]:
    """Return the pull request's title and the issues it closes.

    ``closingIssuesReferences`` is GraphQL-only -- no REST field carries it --
    which is why this posts a query rather than reusing the plain ``GET`` the
    last-push-approval check makes do with.

    Raises ``LinkSetUnreadable`` for every answer that is not a complete link set,
    including a truncated page: a link set cut off at ``LINK_PAGE_SIZE`` would
    report a linked issue as unlinked, and the one thing this check must never do
    is invent a dropped claim.
    """
    owner, _, name = repo.partition("/")
    if not owner or not name:
        raise LinkSetUnreadable(f"repository {repo!r} is not in owner/name form.")
    payload = _post(
        f"{API_ROOT}/graphql",
        {"query": _QUERY, "variables": {"owner": owner, "name": name, "number": pr, "first": LINK_PAGE_SIZE}},
        token,
    )
    if not isinstance(payload, dict):
        raise LinkSetUnreadable("the API response was not a JSON object.")
    if payload.get("errors"):
        raise LinkSetUnreadable(f"the API returned errors: {json.dumps(payload['errors'])[:400]}")
    pull = ((payload.get("data") or {}).get("repository") or {}).get("pullRequest")
    if not isinstance(pull, dict):
        raise LinkSetUnreadable(f"no pull request {repo}#{pr} in the response.")
    references = pull.get("closingIssuesReferences") or {}
    nodes = references.get("nodes") or []
    total = references.get("totalCount")
    if isinstance(total, int) and total > len(nodes):
        raise LinkSetUnreadable(f"the link set is truncated ({total} links, {len(nodes)} read).")
    numbers = tuple(int(node["number"]) for node in nodes if isinstance(node, dict) and "number" in node)
    return str(pull.get("title") or ""), numbers


def render(verdict: Verdict, repo: str, pr: int, title: str) -> str:
    """Render the job-summary report for this run."""
    escaped = title.replace("|", "\\|")
    lines = [
        "## Closing reference",
        "",
        f"Outcome: **{verdict.outcome}**",
        "",
        verdict.summary,
        "",
        "| field | value |",
        "|---|---|",
        f"| pull request | {repo}#{pr} |",
        f"| title | {escaped} |",
        f"| claimed by the title | {_issues(verdict.claimed)} |",
        f"| linked by the pull request | {_issues(verdict.linked)} |",
    ]
    if verdict.is_finding:
        lines += [
            "",
            "### What clears this",
            "",
            f"1. Put the keyword in the **body**: a line reading `Closes {_issues(verdict.unlinked)}`.",
            "   GitHub creates the link when the description is saved, and this check re-runs on",
            "   `edited`, so this is self-clearing without a push.",
            "2. If the pull request should *not* close the issue, reword the title so it does not",
            "   read as a claim -- `per #N`, `follow-up to #N`, `towards #N`. The reference still",
            "   renders and the issue's timeline still shows the cross-reference.",
            "",
            "Note that a keyword inside a code span or a fenced block does not link either.",
            "#1894's body contains `` `closes #1891` `` in backticks and linked nothing, which is",
            "why this check reads the link set GitHub publishes rather than the body's text.",
        ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument("--pr", default=os.environ.get("PR_NUMBER", ""))
    parser.add_argument("--title", default=os.environ.get("PR_TITLE", ""))
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN", ""))
    args = parser.parse_args(argv)

    if not args.repo or not args.pr:
        parser.error("--repo and --pr are required (or set GITHUB_REPOSITORY and PR_NUMBER)")
    if not args.token:
        parser.error("--token is required (or set GITHUB_TOKEN)")

    pr = int(args.pr)
    title = args.title
    linked: tuple[int, ...] | None
    detail = ""
    try:
        api_title, linked = resolve_pull_request(args.repo, pr, args.token)
        title = title or api_title
    except (LinkSetUnreadable, urllib.error.URLError, urllib.error.HTTPError, ValueError, KeyError) as exc:
        linked, detail = None, str(exc)
        print(f"check_closing_reference: {detail}", file=sys.stderr)

    verdict = classify(title, linked, detail)
    report = render(verdict, args.repo, pr, title)
    print(report)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(report + "\n")

    if verdict.is_finding:
        print(
            f"::error title=A closing keyword in the title links nothing::{verdict.summary} "
            f"Move it into the pull request body."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
