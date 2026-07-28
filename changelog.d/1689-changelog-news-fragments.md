### Quality: changelog entries are recorded as news fragments

A behavioural change now records its changelog entry as its own file under
`changelog.d/` (`<pr-number>-<slug>.md`) instead of appending to the
`## [Unreleased]` section of `CHANGELOG.md`. `scripts/assemble_changelog.py`
folds the accumulated fragments into the log when a tag is cut.

Appending to `[Unreleased]` put every branch at the same insertion anchor, so
any two pull requests open at once conflicted the moment either one merged - on
ordering alone, never on meaning. One merge to `main` was observed turning five
already-approved PRs `CONFLICTING`, with `CHANGELOG.md` the only conflicting
file in all five; because stale approvals are dismissed on push, each resolution
cost a re-approval round that reviewed no changed behaviour, and the cost scaled
with the number of open PRs.

A `CHANGELOG.md merge=union` attribute does not address this: local git honors
the driver, but GitHub's mergeability computation does not apply merge drivers,
so the pull request still reports as conflicting and the resolving push still
dismisses the approval. A fragment is its own path, so no two pull requests
touch the same file and there is nothing for a merge to reconcile.

`CHANGELOG.md` remains the human-facing log and is still edited directly for
release bookkeeping. Assembly refuses wholesale if any pending fragment is
malformed or misnamed, so a fragment cannot be silently dropped, nor deleted
without its content landing in the log.
