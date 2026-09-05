### Fixed: `check_merge_blockers.py` names an already-merged pull request instead of asking for a re-read that cannot settle

A pull request that had already merged reported as `merge-state-unknown`, owed by
nobody, with the remedy "Re-read the pull request: mergeability is computed on
demand and settles on a later read". It never settles - a merged pull request is
closed, so `mergeable` stays null for good, and the report described a wait with
no terminating condition while the merge had already landed. Measured on #3219
and #3230 minutes after they squashed: both `closed` / `merged: true` /
`mergeable: null`, byte-identical in every field the check read to open #3205's
genuine recompute at the same moment. The `merged` key was already in the payload
`resolve_state` fetches and was simply not read. There is now a terminal
`already-merged` outcome, reported ahead of every ruleset rule and alone, because
"0 of 1 approvals" is not a rule the merge left unsatisfied. An open pull
request's null still reads as an uncomputed mergeability, unchanged.
