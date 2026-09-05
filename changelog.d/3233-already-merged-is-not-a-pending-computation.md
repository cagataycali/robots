### Fixed: an already-merged pull request is no longer reported as a mergeability still being computed

`scripts/check_merge_blockers.py --pr N` reported a pull request that had
already merged as `merge-state-unknown`, owed by nobody, with the remedy
"Re-read the pull request: mergeability is computed on demand and settles on a
later read". No later read settles it -- a merged pull request is closed, so
`mergeable` stays null permanently, and #2586 still read `null`/`unknown`
fourteen days after it squashed. The advice describes a wait with no
terminating condition, and it reads in the reassuring direction: "no party owes
an action" is literally true, which is exactly what a caller polling for "can I
merge yet" sees while the answer is "you already did".

`mergeable is None` was ambiguous between "GitHub is still computing" and
"there is nothing left to compute", and the genuine transient #2585 named is
live at the same time: measured immediately after #3219 and #3230 squashed,
both merged pull requests and open #3205 were identical in every field the
script read. The field that separates them was already in the payload --
`resolve_state` fetched the whole REST pull request object for five keys, and
`merged` came back in the same response.

`PullRequestState` now carries `merged`, and a terminal `already-merged`
outcome is reported ahead of every rule and short-circuits them: once the
change is on the base, "0 of 1 approvals" is not an unsatisfied rule but a
question about a closed pull request. It is owed by nobody, is deliberately not
a finding so the exit status keeps its meaning, and carries none of the
report's deferral language. `primary` and `_next_action` gained a terminal tier
ahead of the gating one, so a merged pull request can no longer be told "the
answer is not in yet". Reading a null as clean is still wrong (#1035) and the
open recomputing path is unchanged.
