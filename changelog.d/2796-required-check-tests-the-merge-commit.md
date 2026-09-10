### Fixed: the required check evaluates the tree that would merge, not the branch head

`call-test-lint / Test and Lint` was handed `pull_request.head.sha`, a tree that
never exists on `main`. A branch that forked before a grader fix landed on `main`
was therefore red on its own head while the merge it proposed was green, and the
only way to clear the red re-ran the check and dismissed the approval on a branch
that needed no change (#2796, measured on #2789 and again on #3236). The reusable
workflow now checks out `refs/pull/<n>/merge` on a pull request, the same commit
`lockfile-parity` reads for the same reason, and `github.sha` on a push, where no
merge commit exists. `merge-base-overlap` keeps reading the head on purpose: its
merge base against the merge commit would be the base tip and the check would pass
vacuously, which its own comment already says.
