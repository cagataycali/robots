### Fixed: duplicate work is collided on the identity a created path declares, so two changelog fragments naming one change are reported

`scripts/check_duplicate_claim.py --all-open` pairs two open pull requests on a path they both
**create**. A changelog fragment is named `<number>-<slug>.md`, so two branches describing one
change write one slug under two numbers and their paths differ -- and a pair whose only shared
creation is that entry collides on nothing. #2766/#2767 are exactly that pair: both wire
`G1Driver.send_action` to publish on `rt/lowcmd`, opened 8m 06s apart, one putting its tests in a
new file and the other into an existing one, so they create no common path at all. Run against the
open set while both sat in it, the sweep reported `unique-additions - no two of the 9 open pull
requests create the same file`.

The exclusion was not an oversight but an assertion. A test pinned that a changelog fragment can
never be the shared path, with the reason "its name embeds the number, so it cannot collide", and
the fixtures carried a fragment to demonstrate it. The reason is true of the raw path and is exactly
why the number is the half that has to go.

The key is now the identity a created path declares. For every path but a fragment that is the path
itself, so this is a strict widening: over the 2002 pairs in #2345 through #2767 that were open at
the same instant it selects the same 2 pairs the raw path selects, plus #2766/#2767, and loses none.

    what both create                                        pair           closed
    tests/test_recorder_counters_track_on_disk_frames.py     #2388, #2389   #2389
    tests/training/test_checkpoint_cadence_domain.py         #2707, #2708   #2707
    changelog.d/*-g1-send-action-wired.md                    #2766, #2767   open

The stated fear behind the old boundary was that keying on a fragment "would fire on every pair in
the queue". It does not. Of those 353 pull requests, 350 add a fragment and between them use **350
distinct slugs**; exactly two slugs are used twice, and both times the two users are a duplicate
pair. The number is the noisy half: it is meant to be the pull request's own number and in 40 of
those 350 it is not, because it is chosen before the pull request exists and races with whatever
merges first.

The report names the two branches' shared subject as `changelog.d/*-<slug>.md` rather than a
filename, because the file it would otherwise print exists on neither branch. What a fragment *is*
comes from `scripts/assemble_changelog.py` rather than a second copy of the pattern, so this sweep
and the assembler cannot disagree; a name the assembler would reject keys on itself, which can only
fail to report a pair rather than invent one.

The two keys remain complementary. #2707/#2708 is still reachable only from the path -- #2707 added
no fragment at all -- and neither issue-keyed pair in the window shares a created subject, so no
relation may be deleted in favour of another.
