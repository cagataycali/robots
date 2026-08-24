### Added: a check for a checkout the pull request's branch has moved past

`scripts/check_checkout_is_pr_head.py` compares the commit a local clone is on
against the tip of the branch in the head repository, and exits 1 when the branch
has commits the clone does not, so it can sit in front of the work.

A pull request has three answers to "what is the head commit", not the two
`check_pr_head_is_current.py` compares. The third is `refs/pull/N/head`, which is
what a checkout reaches for and the only one carrying no signal: it is a mirror
ref GitHub refreshes on its own schedule, so it trails a push to the fork
branch. Measured on #2678, where the mirror served `33f8bcf4` while the API
already served the tip `0b070a05`, one commit further on and with the mirror as
its ancestor - a pure lag.

The cost is not a wasted fetch. The run believed it had checked out the branch,
grepped the symbol its review thread named, found it genuinely unused on that
tree, and derived a correct fix for source that no longer existed; only `git
push` refusing as non-fast-forward surfaced the drift, after the work. The answer
was also worse than a duplicate. The thread was a CodeQL "unused global", which
against the stale tree means deletion - 18 passed, `ruff` clean - and against the
tip means the opposite, that the constant was unused because the behavioural
tests spelled its value out at each call site and the fix is to wire it up.
Deletion would have passed CI while removing the invariant the symbol carried.

The comparison is ancestry, not equality. A clone sitting at its own unpushed
commit contains the tip and reads `ahead`, which is the ordinary state between a
commit and its push and is not a finding - a check that reported it would fire
exactly when it is being used correctly. A tip absent from the local object
database reads `stale-checkout` rather than indeterminate, since a clone that
never fetched a commit cannot contain it.

The tip is resolved from the head repository's own ref, because both cheap
answers are ones this check exists to catch out. Its remedy is to fetch the
branch by name and re-read the pull request's threads against that commit, never
to rebase or amend onto it: the commit the branch already carries may be the
answer to the thread, and #2520 records a plain push being refused as the only
thing that prevented its destruction.
