### Added: `check_merge_base_overlap.py --all-open` sweeps open pull requests for pairs editing one file

The merge-base overlap check compares one branch against its base, which is a
property no pair of *open* pull requests has: while both are open neither is in
the other's ancestry, so the check reads green on both and says nothing. The
property lives between two pull requests, and a check scoped to one branch
cannot hold it -- which is how #1766 and #1763 landed hours apart and turned
`main` red, and how #2233 and #2235 came to compose to a red suite while each
reports clean.

`--all-open` enumerates the pairs. Measured over the 19 open pull requests on
2026-08-13: 171 pairs, 4 sharing a behaviour-bearing file. It reports and asks
nothing of either author -- whichever merges first is unaffected, and the second
then gets the base overlap after the fact. Pairs sharing only documentation are
listed and not counted, reusing the single-branch mode's own prose rule so the
two cannot drift. A pull request whose file list could not be read completely is
named as not evaluated rather than compared on a partial set, because the shared
path may be on the page nobody read and folding it in would report the
reassuring answer.
