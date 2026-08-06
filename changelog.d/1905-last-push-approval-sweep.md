### Fixed: the last-push-approval report can be pointed at the standing open pull requests

`scripts/check_last_push_approval.py` gained `--all-open`, which classifies every
open non-draft pull request in one pass instead of one named pull request.

The check was reachable only from `pull_request` and `pull_request_review`
events, so it could evaluate a pull request only if one had fired *since the
workflow landed* - and the population it was written for is the population that
had not. #1035's head was pushed 2026-08-01 and approved 51 minutes later, both
before the workflow existed on 2026-08-04, so `Detect an approval the last pusher
cannot supply` is absent from the 11 check runs on that head. The verdict was
never wrong: run directly, the same script reports `pusher-only-approval` on
#1035 and #1722 immediately. Nothing was asking it.

A sweep is what a status scan needs to tell `awaiting-first-review` from
`pusher-only-approval`, since `reviewDecision: REVIEW_REQUIRED` with
`mergeStateStatus: BLOCKED` describes both and they need opposite actions - the
conflation that stood in eight consecutive scan summaries as "reviewer bandwidth
is the sole constraint".

Drafts are excluded, because a draft cannot merge whatever its approvals say. A
per-pull-request lookup failure is named in the report and skipped rather than
allowed to abort the sweep, so one rate-limited pull request cannot suppress a
finding on another or read as a clean sweep. Neither `--all-open` nor `--pr` may
be silently ignored when both are given; the invocation is refused.
