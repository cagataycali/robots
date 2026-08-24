### Added: a closing keyword that only appears in a PR title is refused, not silently dropped

GitHub parses closing keywords from a pull request's body and from its commit
messages, never from its title. A title that ends in a closing keyword and an
issue number therefore links nothing, and nothing on either side of the merge
reports it: a bare cross-reference renders identically to the start of a closing
link, so the title reads as a claim to every human who sees it while the field
that would contradict it is one nobody opens.

Measured over the last 100 pull requests here - 29 titles carry a closing keyword
before an issue number, 27 also linked the issue, and two did not. #1894 claimed
issue 1891 in its title alone, and that issue was still open two days later;
#1923 claimed issue 1912 the same way and it had to be closed by hand. #1961
counts the wider cost, 18 of the last 30 merges having neither a board item nor a
closing link, and asks for this check by name. This is that half of it; the board
question needs a decision and stays open there.

`scripts/check_closing_reference.py` compares the numbers a title claims against
`closingIssuesReferences`, surfaced by `.github/workflows/closing-reference.yml`.
It deliberately does **not** scan the body, because that implementation passes the
incident it was written for: #1894's body does carry the keyword, inside a code
span, which GitHub does not link - so a text scan and GitHub disagree on precisely
the pull request that matters. The link set is the answer GitHub already
publishes, so the only text parsed is the title, where GitHub does nothing and
there is consequently no ground truth to read. That the Actions token reads that
field truthfully was verified against three pull requests rather than assumed,
since #1961 records `projectItems` returning a false `0` under the same token; an
unreadable or truncated answer is reported as its own outcome and passes, so a
permission change can turn this neither into an accusation against every branch
nor into a silent no-op.

Four outcomes, because a title that claims nothing (71 of 100) and one whose claim
is linked (27) are both ordinary and already visible. Only the two that dropped a
claim are a finding. Unlike the last-push-approval report this one is
self-clearing - moving the keyword into the body creates the link and the check
re-runs on `edited` - which is why it fails rather than warns, and why it
subscribes to an activity type that changes no code: its inputs are the title and
the link set, and that is the only event that changes either. The prohibition in
`tests/test_pull_request_trigger_types.py` is narrowed rather than waived, since
the harm it measures needs the required check to be started by the event, and the
two properties that make the exemption safe are now pinned there.

What a title already claims is all this reads, so it needs no answer to whether
every change must trace to an issue - the separate question #1961 raises. A pull
request that claims nothing anywhere stays uncaught, which is the other way a link
is lost.

Tooling, tests and documentation only; no production code or runtime behaviour
changes.
