### Fixed: a standing request for changes is named as its own merge blocker

`scripts/check_merge_blockers.py` modelled only the approval side of a pull
request's review decision, so a pull request sitting at `CHANGES_REQUESTED` was
reported as `missing-approval`, owed by `a reviewer other than the pusher`.
That party cannot clear it: with required reviews in force a standing request
for changes holds the merge until **its own author** approves or dismisses it,
so another account's approval satisfies `required_approving_review_count` and
the pull request stays `BLOCKED`. The report therefore named a reviewer whose
approval could not have merged it -- the presentation issue #1905 records,
reached from the review-decision side rather than the last-push side.

Every other sweep reads clean on the shape, which is what made it expensive.
Measured on #3205: `reviewDecision` `CHANGES_REQUESTED` and `mergeStateStatus`
`BLOCKED`, with its one review thread **resolved**, `call-test-lint` SUCCESS,
`check_pr_head_is_current.py` reading `current`, and
`check_thread_is_answered.py` reading `nothing-owed`. Thread resolution and the
review decision are separate objects, so resolving the thread does not retract
the review and the thread sweep was right to say nothing was owed; the
requester's own follow-up reply is a `COMMENTED` review, which expresses no
position and supersedes nothing. It stood 15h44m, of which 12h51m was after the
fix had landed and the thread was resolved.

The new `changes-requested` outcome is owed by `the reviewer who requested
changes`, names each holding account in its detail rather than only the role,
and is reported ahead of the approval rules it is not answerable by. It is
scoped to a ruleset that actually requires approving reviews, for the same
reason the pusher discount is, and is deliberately neither gating nor a finding:
a failing required check beside it is still independently the author's to fix,
and an outcome that fires whenever a review is in progress fires on the ordinary
state.

Standing is resolved by `check_last_push_approval.py`, which gains
`current_change_requesters` beside `current_approvers` over one shared
`_latest_positions`, so the rule that a `COMMENTED` review retracts nothing
cannot hold for approvals and lapse for requests. The sweep binds that resolver
rather than deriving standing itself, as it already did for approvals.

`AGENTS.md` gains `CHANGES_REQUESTED` as the fourth reading of `reviewDecision`
beside the existing `null`, `APPROVED` and `REVIEW_REQUIRED` ones, and the
enumeration of parties the sweep names gains this outcome so the file cannot
drift from the tool.
