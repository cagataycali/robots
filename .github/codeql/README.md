# CodeQL configuration

This directory contains the CodeQL configuration consumed by both
`.github/workflows/codeql.yml` and `.github/workflows/codeql-advanced.yml`
via the `config-file:` key on the `github/codeql-action/init` step.

## Files

- `config.yml` -- query filter rules. See inline comments for per-rule
  rationale.

## Suppressed queries

### `py/unsafe-cyclic-import` (global)

**Why it fires:** the rule walks `if TYPE_CHECKING:` blocks for cycle
detection. `strands_robots/simulation/{base,policy_runner,benchmark}.py`
form a deliberate static-only cycle so `policy_runner` can call into the
`SimEngine` ABC defined in `base`, while `base` can advertise
`PolicyRunner`-typed return values to static checkers (mypy, IDE
navigation, `typing.get_type_hints` consumers).

**Why it is safe at runtime:** every file uses `from __future__ import
annotations` (PEP 563). All type hints are string-form at runtime and
are never resolved at module-import time. The lazy-import shim
`_lazy_policy_runner()` defers the actual `PolicyRunner` / `VideoConfig`
class lookup to call time, by which point both modules are fully
loaded. Tests pin the runtime invariants:

- `tests/simulation/test_no_cyclic_imports.py` -- spawns a fresh Python
  interpreter for each of the four affected modules and asserts each
  imports cleanly with no recursion error.
- `tests/simulation/test_no_import_cycle.py::test_no_runtime_import_cycles`
  -- builds the runtime import graph (excluding `TYPE_CHECKING` edges)
  via `networkx.simple_cycles` and asserts it is acyclic.

If the runtime cycle is ever reintroduced (e.g. by hoisting a lazy
import to module scope), those pins fail loudly in CI -- independent of
CodeQL's view of the AST.

**Why a global suppression rather than path-scoped:** CodeQL's
`query-filters` keys filter by `id` / `tags` / `precision` only;
path-scoped exclusion of a single query is not supported (the only
path-aware key is `paths-ignore`, which excludes a file from all
queries -- too broad). The simulation triple is the only file shape in
the repository where this query fires today, so a global exclude is
equivalent in effect and keeps the config small. A future contributor
who introduces a *new* legitimate violation in unrelated code would
still want to know about it; in that case, drop this suppression and
fix the new cycle properly.

**References:**

- PR #209 -- the multi-round attempt to satisfy CodeQL via code surgery.
  Paused at draft after the constraint conflict between mypy and
  `py/unsafe-cyclic-import` was confirmed locally; full analysis in
  the S13/R6 changelog of that PR.
- Issue #215 -- this follow-up tracking the suppression.
- CodeQL alerts #83, #84 -- closed by PR #209 R4.
- CodeQL alerts #85, #86, #87 -- in `benchmark.py:42` and
  `policy_runner.py:49-50`; closed by this suppression.
- CodeQL alerts #253, #254, #255 -- recurrent alerts on `base.py:28`
  introduced by PR #209 R5 attempting to recover mypy type info; closed
  by this suppression.

## How to extend

When adding a new suppression:

1. Add the `query-filters` entry to `config.yml` with a thorough
   inline comment explaining the rule, why it fires here, and why
   the suppression is safe.
2. Add a section to this README mirroring the inline comment, plus
   the regression contract (pin tests, runtime-safety argument).
3. Ensure the affected source files carry a top-of-file comment
   pointing at the config, so future readers find the rationale
   without grep-spelunking.
