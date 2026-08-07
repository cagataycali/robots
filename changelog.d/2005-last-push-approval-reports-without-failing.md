### Fixed: the last-push-approval report no longer makes a healthy branch look broken

`last-push-approval.yml` names a pull request whose only approvals come from the
account that pushed its head, and it reported that finding through its exit
status. A single non-`SUCCESS` context drags `statusCheckRollup.state` to
`FAILURE`, and the rollup carries no reason, so the red was indistinguishable
from the branch's own tests failing. Measured on #1722: every required context
`SUCCESS`, threads resolved, `mergeable: MERGEABLE`, rollup `FAILURE` with this
report as the only non-`SUCCESS` context - misread as a broken diff four times.

The finding now leaves the job green and lands where it already landed before
this change: the full report in `$GITHUB_STEP_SUMMARY` and a `Needs an approver
who did not push the head` annotation. The script's own exit status is unchanged
at `1`, because `--all-open` sweeps are consumed that way; only the workflow's
adoption of it as a job conclusion changed. A status that is not 0 or 1 still
fails the job, so red now means the check could not compute an answer rather
than that the branch needs another human. The check row is renamed to `Report
the last-push-approval state`, since green must not assert the absence of a
finding.
