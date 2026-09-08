### Fixed: check_merge_blockers names whether auto-merge will perform the merge it is waiting on

`scripts/check_merge_blockers.py` reported `required-check-pending` as owed by
nobody and `no-unsatisfied-rule` with the remedy "attempt the merge", both of
which assume a person merges once the rules are satisfied. On this repository
29 of the last 30 merged pull requests had auto-merge armed and merged
themselves the second the required check concluded, so a scheduled pass that
read the report and polled the check in order to merge spent twenty minutes
waiting to perform a merge that was never its to make.

The report now reads `auto_merge.enabled_by` from the pulls payload it already
held, prints an `auto-merge` row on every pull request and a column on the
sweep, and on the two waiting outcomes adds a note saying GitHub performs the
merge and that `merged` is the field to re-read first. No outcome, owner or
exit status moves: auto-merge is not a rule.
