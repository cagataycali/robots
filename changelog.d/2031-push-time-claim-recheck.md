### Docs: the duplicate-claim question is asked again before the first push

Step 1's duplicate-claim read is a claim about minute 0. Nothing asked it again, and
authoring a tested change takes longer than the window in which a collision becomes
observable -- every pair measured for #2017 opened inside one ~35-minute span, and a
cycle that took #2029 at intake with a clean answer found it claimed, approved and
merged roughly forty minutes later. What a stale intake read costs is the authoring
and, had the push landed, a second review approval spent arguing preference over an
already-closed defect. Step 5 now asks for the second read, and says which fields
answer it, because the two cheap forms are each insufficient in the reassuring
direction.

Re-running step 1's command is only correct while the rival is still open.
`check_duplicate_claim.py` reads `repository.pullRequests(states: OPEN)`, so a rival
that merges leaves the set and the verdict returns to `unique-claim`: #2030 opened
07:24:45 and merged 07:43:54 closing #2029, and the same command at 07:56 reported
`unique-claim` with exit `0` over four compared pull requests, while #2029 was
`CLOSED` / `COMPLETED` with #2030 recorded as its closer. That is nineteen minutes of
visibility inside a forty-minute authoring window. The issue's own `state` and
`stateReason` do not move, so they are the read that stays true; the command is the
one that names the rival, which is why both are asked rather than one replacing the
other.

For a review-round push the local form has the same shape of hole. Comparing the
branch against a sha recorded at the start catches a sibling push, but a squash merge
writes a new commit onto the base and never moves the head ref, so the comparison
passes across a merge. On #2015 that cost a round: merged 23:13:13 with `headRefOid
ea5e3ff8` against `mergeCommit 1026088`, and a round pushed a minute earlier left the
fork branch at `e7ab4d5b`, which is not an ancestor of `main`. The comparison passed,
the push succeeded, and the content was orphaned; the recovery was a second pull
request. `pullRequest { state mergedAt }` is the field that sees it, and the same read
carries `reviewThreads` at no extra cost -- an unresolved thread is also only
unresolved as of the read, and on #2028 one arrived sixteen minutes after the commit
that was pushed before it.

Guidance rather than a check, by necessity rather than by choice: the collision is
between an unpushed local tree and a remote pull request, which no workflow can
observe. The script's deliberate silence about whether an issue is closed is left
alone -- refusing a pull request for that would accuse correct work whose issue
someone else closed first -- and the difference is the same mode split as its `--repo`
default, since what is decisive for unpushed work is not what a review check should
refuse. `tests/test_push_time_claim_recheck.py` drives the production classifier over
the recorded link sets from both sides of the #2030 merge, so the pin is that the two
reads disagree about one moment rather than that the file mentions them.
