### Fixed: two branches are collided on a module one of their tests names, not only on a path they share

`scripts/check_merge_base_overlap.py` intersected changed **paths**. `main` went red at
`828f80eb` from #2762 and #2774 with that intersection not merely unhelpful but empty. #2762
added `tests/drivers/test_reachy_transport_guards_are_reachable.py`, whose fixture evicts
`"strands_robots.device_connect"` from `sys.modules` and patches
`"strands_robots.device_connect.reachy_transport.api"` by string; #2774 rewrote
`strands_robots/device_connect/__init__.py` to resolve its names lazily. Six of that file's nine
tests failed on the composition, five of them standalone, and the two branches changed no path in
common -- so the sweep reported `no pair in the open set shares a changed path` while both sat in
the queue, and the first tree in which the two were compiled together was `main`.

The exclusion was reasoned rather than accidental, and the reasoning covered a different case.
The file already measured and rejected widening the path set to a test's *walked root*, on the
grounds that a relation firing on 11 of 36 pairs and naming no defect reads as boilerplate. That
argument is about a population a grader never names. A test that reaches into a module *by name*
writes the coupling down, and the name is not a path.

There is now a second relation over the same two path sets and one further input: the dotted
module names a pull request's test diff writes as string literals, resolved to the module files
they name. Every prefix resolves, because importing `a.b.c` executes `a/b/__init__.py`, and that
prefix is two segments shallower than the literal #2762 wrote -- it is where the #2774 edit was.
Measured over the 2676 co-open pairs in #2309 through #2792, the path relation selects 104 pairs
and this one selects 25, of which 9 are not already reported: 0.34% of the population against the
rejected widening's 31%. The nine name couplings a reader can act on -- #2774 + #2762 is the
composition above, #2767 + #2762 and #2767 + #2750 are three driver branches contending for
`strands_robots/drivers/__init__.py`'s registration table, and #2546 + #2545 pairs a branch
removing an export from `strands_robots/mesh/__init__.py` with one whose test names that package.

The bare package root is deliberately not resolved: including it adds four findings, all of them
pairs with one branch's edit to `strands_robots/__init__.py`, which every literal in the tree
names by its first segment. Literals are read from the `patch` field of the entries the path set is
already built from, so the sweep makes no additional request, and both modes hand `(path, patch)`
pairs to one extractor so neither can disagree with the other about what a literal is. What stays
out of reach is unchanged and is still the filesystem walk, which the report continues to name.
