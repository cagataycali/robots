### Fixed: the last-push-approval check no longer names a run's approver as its pusher

`scripts/check_last_push_approval.py` read `triggering_actor` from the workflow
runs on the head sha. That field names the account behind the *latest attempt*,
so GitHub rewrites it when a maintainer **approves a held run** or **re-runs**
one; `actor`, the account whose event created the run, is the pusher. Every run
on a first-time contributor's fork starts at `action_required`, which makes
approving them a maintainer's ordinary first act -- and the check then reported
that maintainer as the pusher of a branch in a repository they cannot push to.
Measured on the same nine `pull_request` runs of #3448, head `b3d2233a`:
`actor: shipitfast`, `triggering_actor: cagataycali`. Filtering on the event does
not help, because the approval re-attributes the existing `pull_request` run
rather than creating a new one.

The cost landed on the population where first-review latency matters most: once
the approving maintainer approved the pull request, the check read
`pusher-only-approval` and `check_merge_blockers.py` reported the pull request as
owed to "a reviewer other than the pusher", while GitHub's own
`require_last_push_approval` rule -- which reads the real pusher -- was satisfied
and the pull request was mergeable. That is the #1905 presentation reached from a
further cause.

`resolve_pusher` now reads `actor`, keeping the event filter, which stays
load-bearing: a `pull_request_review` run's `actor` is the reviewer. Sweeping the
open pull requests, three rows change to the contributor who pushed them and the
rest are unchanged. `actor` agrees with `triggering_actor` on every case this
check already pinned (#1894, #1920, #1722, #1035 and #1921's own), including the
two heads whose commit metadata does not answer at all, so nothing the old field
got right is lost.
