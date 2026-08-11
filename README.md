# newton-seed-refusal — measurement artifact

* `capture.py` — measures the 13-cell cross-backend refusal matrix. The **before**
  matrix is read from a full-suite `--cov` JSON on unmodified main (a two-file
  subset under-reports cells other modules cover); the **after** matrix applies
  coverage monotonicity and asserts exactly which two cells the new class closes.
* `mutate.py` — the mutation table. Each anchor is scoped to its enclosing
  function by AST (`in_fn` / `in_file` printed as the justification) and the
  source is asserted byte-identical on restore.
* `compose.py` — renders the figure. Every rendered number is asserted against
  `facts.json` before the PNG is saved, plus a text-placement guard and a
  per-side white-border check.
* `facts.json` — the measured data both the figure and the PR body quote.

Tests only: no policy, simulation, rendering, recording or asset behaviour
changes, so the artifact is the coverage-and-mutation measurement rather than a
rollout.
