### Fixed: a duplicate pair that claims no issue is now reported, keyed on the file both branches create

`scripts/check_duplicate_claim.py` had one key, `closingIssuesReferences`, and both of its modes
read it -- `--issue` at intake, `--pr` at review. Measured over the last 300 pull requests (#2345
through #2708), **249 of them link no issue at all**, so for that 83% of the traffic there is no
key to collide on: both modes report a unique claim while looking straight at a duplicate pair.
Two of the four duplicate pairs in that window were exactly that shape, and each spent a full
authoring plus review round on a change that could never ship:

    shared added path                                       pair           closed
    tests/test_recorder_counters_track_on_disk_frames.py    #2388, #2389   #2389
    tests/training/test_checkpoint_cadence_domain.py        #2707, #2708   #2707

The residual was recorded in the module as out of scope, justified by one measurement: 18 of the
last 30 merges would fail a rule *requiring* a claim. That measurement stands and the conclusion
drawn from it was wider than it supports -- it rules out demanding a claim, which neither
claim-keyed mode does, and says nothing about colliding the pair on a different key. A changed-path
set exists for every branch whether it claims anything or not.

A third mode, `--all-open`, reports a pair whose **added** paths intersect. The narrowness is the
whole claim: over the 1802 pairs that were open at the same instant, the added-path relation
selects **2** and both are duplicates, where intersecting *any* changed path selects 117. Those 117
are a composition question with a different remedy -- a merge order, possibly one test run -- and
`scripts/check_merge_base_overlap.py --all-open` already owns it; two branches *creating* one file
is not a composition to verify but two answers to one question. For contrast, the path relation
this repository has already rejected (widening to a test's walked root) selected 11 of 36 pairs and
named no defect.

The two keys are complementary rather than nested, so neither replaces the other: neither
issue-keyed pair in that window shares an added path, and neither claim-free pair claims an issue.
Four duplicate pairs, two reachable from each key, none from both. The new mode also keeps this
file's own policies rather than the sibling sweep's -- drafts are included, because a draft's new
file is authored work whatever its merge state, and the open set is read through
`repository.pullRequests(states: OPEN)` rather than `search`, which is eventually consistent and
would miss a pull request opened seconds ago. That is the row this sweep exists to find: both
claim-free pairs opened inside the same ~35-minute window every other observed collision shares,
14m 41s and 29m 26s apart.

It cannot be asked at intake, and the module now says so instead of leaving it implied: a path set
is a property of a pushed branch, so there is nothing to read before the work starts. `--all-open`
caps the review cost of a collision, which is what `--pr` does for a claim, while `--issue`
remains the half that prevents the authoring. A truncated file list, an unreadable open set and an
API error all reach `unknown-additions` rather than a pass, for the reason the claim-keyed modes
refuse a truncated link set: a sweep that reports clean because it could not look far enough is
worse than no sweep, because it looks like one.
