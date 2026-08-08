### Docs: the composition shortcut's compare ref is fork-qualified, and its post-merge half is scoped

Step 8's cheap read for whether a pre-merge composition is owed - `behind_by` on a
comparison against `main` - was documented as `compare/main...<head>`. That ref does
not resolve in the base repository, and step 1 mandates that the branch live on a
fork, so the documented form returned `404` for every pull request here. Same head,
same instant: `main...feat/ackermann-ros-robot` answers `404 Not Found` while
`main...Vivek0712:robots:feat/ackermann-ros-robot` answers `diverged ahead_by=11
behind_by=116`.

The consequence was the loss of the whole saving the field was introduced for. A
reader who gets a `404` reaches the clause saying a head that cannot be compared is
not a `0`, and runs the composition - a clone and two suite runs - on every branch,
which is exactly what reading `behind_by` was meant to avoid. Nothing contradicted
it, because a `404` followed by a composition run looks like diligence and the
composition then confirms nothing is wrong. The passage now gives the qualified
template, says to resolve its parts from `headRepository { nameWithOwner }` and
`headRefName` rather than assume them, and notes that the same form serves a branch
in the base repository, so there is one spelling rather than a choice. The
uncomparable case is narrowed rather than dropped: a `404` has two causes wanting
opposite actions, and only a head sha that is genuinely gone is the one the earlier
clause describes.

The same step's post-merge half gains the scope it was measured under. Comparing
`.commit.tree.sha` between the head CI went green on and its squash on `main` is
evidence only when `behind_by == 0`; when the branch is behind, the squash tree
incorporates the intervening commits, so the trees differ for a correct merge. #2012
at `behind_by: 0` has equal trees (`e174201b7ccf`), and #2024 at `behind_by: 1` has
`4af91f210d09` against `8b3e7e8a3434` with `main` green on all four checks
afterwards. Stated unscoped, the check invites reading a good merge as drift.

`tests/test_composition_compare_ref_form.py` lifts the ref template out of the prose
and renders it against a recorded pull request head, so the assertion is that the
documented ref is the one the API answered rather than that the file mentions forks -
a template tidied back to `<head>` fails it.
