### Fixed: the test suite collects identically in every process, so it can be run in parallel

`pytest-xdist` has every worker collect the whole tree independently and then
compares the collections: one test ID that differs between two workers aborts
the session with "Different tests were collected between gw0 and gw1" before a
single test runs. Three expressions in two files made the ID a function of the
process rather than of the source - `time.time()` inside a `parametrize` table,
and `object()` / `memoryview(...)` rows labelled `ids=repr`, whose default
`__repr__` prints the instance's address. The clock read is now a written-down
stamp of the same shape, and the two address-bearing rows are labelled by type;
`ids=repr` is untouched everywhere it renders literals, which is the 230 other
sites. The rule is derived over the whole tree by
`tests/test_parametrize_ids_are_the_same_in_every_process.py` rather than pinned
at the two files that broke it, because nothing made either site special.

Measured on one 2-CPU machine, whole suite, same flags: collection aborted after
39 s before, and now collects 55,405 tests and runs them in 19 min 22 s against
37 min 55 s serially (1.97x). Ten cells in four files still fail only under
parallel execution and are not addressed here.
