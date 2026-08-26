### Docs: time the changelog fragment against the review, not the push

`AGENTS.md` step 3 explained why a news fragment replaces a direct
`CHANGELOG.md` append, but said nothing about the push the fragment convention
itself introduces. The fragment is named `changelog.d/<number>-<slug>.md` with
the pull request's number, which does not exist until the pull request is open,
so a change claiming no issue can only add it as a second commit onto an
already-open branch -- and `dismiss_stale_reviews_on_push` does not exempt it.

Step 3 now records that the push is free only while no approval has landed,
measured over four pull requests where the two that pushed the fragment within
about a minute of opening kept their approvals and the two that pushed it after
an approval each paid a re-approval round. It names the three ways out: push the
fragment immediately, open the pull request as a draft and mark it ready once the
fragment is in, or use the issue number when the change claims an issue, which is
known at intake and needs no second push. It also records that folding the
fragment in with `--amend --force-with-lease` after review has started is what
dismisses an approval rather than what avoids one.
