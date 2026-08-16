### Added: a CI check that refuses a pull request whose base moved under the files it edits

`main` went red at `0e636f8` from two pull requests that were each individually
green and textually non-conflicting. #1766 and #1763 both edited
`_recompile_preserving_state` in `strands_robots/simulation/mujoco/scene_ops.py`,
for unrelated reasons. #1766 landed first, after which every signal the merge
gate offers still read green on #1763 -- `reviewDecision: APPROVED`,
`statusCheckRollup: SUCCESS`, `mergeable: MERGEABLE`, `mergeStateStatus: CLEAN`
-- and the squash broke the suite anyway, because #1763 carried a *premise* test
asserting the exact defect #1766 had just fixed.

None of those four signals could have caught it. They are all computed against
the base the branch was tested on, so the first evaluation of the two changes
together was `main` itself. `mergeStateStatus: CLEAN` is a statement about
**text**: git had no conflicting hunks to report, and it is not git's job to know
that one branch's assertion describes the other branch's bug.

`scripts/check_merge_base_overlap.py`, run by the new
`Merge Base Overlap Check` workflow on every pull request, intersects the paths
the branch edits since its merge base with the paths its base branch changed over
the same span. A non-empty intersection does not prove the combination is broken;
it proves the weaker and still blocking fact that **the combination has never
been compiled**. For the pair above the intersection is one entry, `scene_ops.py`.

The remedy is self-clearing: merging the base branch advances the merge base,
which both empties the intersection and re-runs the checks against a base
containing the landed commits. That is why the check ships with no bypass
switch. An overlap confined to `.md` / `.rst` / `.txt` is reported without
blocking, since prose cannot change what the suite or the package does and a
genuine collision inside one surfaces as a merge conflict.

This is the targeted form of branch protection's "Require branches to be up to
date before merging", which demands an update plus a full ~14.7k-test re-run
before every merge; this asks for one only when the branch and its base actually
edited the same file.
