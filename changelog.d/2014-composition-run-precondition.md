### Docs: a pre-merge composition run is gated on the branch actually being behind

Step 8 prescribed merging `main` into an approved branch and re-running the
affected tests whenever a second approved PR touched a file a just-merged PR had
also touched. The rule is right and the trigger was not: file overlap says two
pull requests touched the same file, not that either one landed outside the
other's ancestry, which is what makes a pair of changes exist that CI never
compiled together. So the expensive branch of step 8 - a clone, a merge and two
suite runs that must be read as a delta - fired on branches with nothing to
compose, and it did not look like a mistake, it looked like diligence.

One field on a comparison already available separates the two cases exactly. The
head that broke `main` reports `diverged  ahead_by=2  behind_by=1` against the
`main` it merged into; the branch that met the same overlap trigger with nothing
to compose reports `ahead  ahead_by=3  behind_by=0`, its overlapping neighbour 13
commits back in its own ancestry. The passage states both directions of the
asymmetry, because only one of them is sound: a `behind_by` of `0` proves nothing
needs composing, while a `behind_by` above zero does not prove a conflict exists,
so the file-overlap read stays as the narrowing that selects which branches are
worth a run rather than being replaced by it. It also records that the counts are
totals rather than page counts - a comparison 877 commits wide reports the true
count beside a `commits` array truncated to 250, and the reverse direction
reports it beside an empty one - and that a comparison which cannot be made at
all is not a zero.

The post-merge half of the same step gains the no-clone form of its check.
Comparing the two commits' tree shas is whole-tree where
`git diff --name-only ... -- strands_robots/ tests/` is path-scoped, and it needs
neither the local composition to still exist nor a suite, which is what the
batching case in the same step otherwise has no evidence for. #2001 is a commit
whose entire diff falls outside both prefixes, so the path-scoped form reports
empty across a real tree change.
