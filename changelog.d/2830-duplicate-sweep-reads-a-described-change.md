### Fixed: the duplicate-work sweep collides two branches on the change they describe, not only on the file they create

`scripts/check_duplicate_claim.py --all-open` paired open branches on what they
create - a path, or the slug of a changelog fragment. That key reads a name, and a
name is what two authors describing one change need not share: #2820 and #2822
fixed one defect thirteen minutes apart and called it
`feetech-broadcast-is-not-a-reply-address` and
`feetech-motor-id-excludes-the-broadcast`, so the sweep reported
`unique-additions` while both were open.

The sweep now asks a second question beside the first: do two branches' fragments
share at least two words *and* do both edit one pre-existing test? Two authors
fixing one defect describe the same subject and correct the same case. The
conjunction is the relation - neither half is usable alone, and the report keeps
the two keys apart because they support different conclusions. A shared created
path is a fact about the branches; a shared description over a shared test is a
pair to read.

Measured over the 2199 pairs open at the same instant in #2345 through #2825,
against the eleven pairs whose closed half names, in its own closing comment, the
pull request that superseded it: the created-path key alone reaches five of the
eleven, the new key reaches nine, and the two together reach ten. Shared words
alone would select 33 pairs and fire on 37.2% of replayed sweeps - the repository
names its changes in a house style, so `names` and `the` collide constantly - and
a shared edited source file, the obvious widening, is 8.8% precise and fires on
29.9%. The conjunction selects 14 pairs at 64.3% precision and fires on 5.6%.

The measurement also corrects a denominator: eleven declared duplicate pairs is
more than the five this module counted before, because a closing comment reaches
pairs neither key found. The created-path key's recall is 45.5% of that set, not
most of it.
