# harness_memory: unusable-input branches

Measurement artifacts for the PR that pins the five branches of
`strands_robots/tools/harness_memory.py` reporting an unusable input or store.

* `census.py` - the coverage census that selected the slice (per-function
  contiguous-run / fraction views, plus the uncovered-refusal-line view that
  found this module).
* `capture.py` - drives each branch and records the reason a caller gets,
  together with its coverage before and after. Writes `art_facts.json`.
* `mutate.py` - the mutation table: six plausible regressions, each applied to
  the branch it targets (AST-scoped to the enclosing function) and run against
  both the new tests and the pre-existing ones, restoring the source
  byte-identically after each. Writes `mutations.json`.
* `compose.py` - builds the figure from those two JSON dumps. Every rendered
  number is asserted against them before the image is saved.

Reproduce: run `capture.py` then `mutate.py` then `compose.py` from the repo
root with `PYTHONPATH` set to it.
