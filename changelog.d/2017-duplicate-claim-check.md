### CI: report whether an open pull request already closes an issue

Every check here reads one pull request at a time, so a branch claiming an issue
another open branch already claims passed all of them. Over the last 100 pull requests
three pairs did exactly that -- #1944/#1946 (#1942), #1995/#1996 (#1994) and
#2015/#2016 (#2007) -- and all three abandoned halves had already been **approved**, so
what a duplicate spends is a review approval on a change that could never ship. Two of
the pairs also carried a real `git merge-tree` content conflict, so they could not both
have landed.

`scripts/check_duplicate_claim.py` reads `closingIssuesReferences` across the open pull
requests. `--issue N` is the intake question, asked before a second pull request exists;
`--pr N` compares an existing one's claims and excludes itself. It takes no position on
which of a pair is at fault: measured on those pairs, the *newer* pull request is the one
that merged in two of the three cases. An unreadable or truncated answer reports
`unknown-claims` rather than a pass, and the open set is read through
`repository.pullRequests(states: OPEN)` rather than `search`, which is eventually
consistent and would return a clean answer inside the ~35-minute window where every
observed collision happened. AGENTS.md step 1 asks the question at intake, which prevents
the duplicated authoring rather than capping it.
