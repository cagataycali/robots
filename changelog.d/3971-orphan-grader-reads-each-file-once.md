### Changed: the orphan grader reads each file once for all three rules

The three rules in `tests/test_sys_modules_removal_leaves_no_orphan.py` each
walked every graded file for the `sys` aliases and the method owners, then
walked every function's subtree another seven times for shapes almost no
function holds - 40,915 functions across the test trees, 300 of which touch
`sys.modules` at all. The cell that pays for the first read was the largest
single cell in the suite (#3869). One breadth-first pass per file now reads the
registries, the owning class, the module-level string bindings and the
functions a rule can report on, and the rules walk only those. Measured with
the trees already parsed, `_readings()` took 49 s before and 12 s after with
identical results file for file. The pre-filter's one failure direction - a
function it skips and a rule would have reported - is pinned per spelling the
rules read.
