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

**Why it is safe at runtime:** the cycle is asymmetric, not closed.

- `simulation/base.py:50` imports `PolicyRunner` and `VideoConfig` from
  `simulation/policy_runner.py` at module level (the runtime edge that
  static analysers see and flag).
- `simulation/policy_runner.py:51-54` imports `SimEngine`,
  `BenchmarkProtocol`, and `Policy` only under `if TYPE_CHECKING:`.
  These are not real edges at import time -- `TYPE_CHECKING` is `False`
  at runtime, the block is skipped, and the names exist only for the
  static type system.
- `simulation/benchmark.py:46-47` similarly imports `SimEngine` only
  under `TYPE_CHECKING`.

There is therefore no runtime back-edge from `policy_runner` (or
`benchmark`) into `base`. `base` finishes importing cleanly before
anything in `policy_runner` runs at module scope, so the cycle never
closes at import time. `from __future__ import annotations` (PEP 563)
on every affected file is the additional belt-and-braces guarantee
that type-hint resolution itself never re-enters `base` or
`policy_runner` via `typing.get_type_hints` consumers.

The CodeQL-independent regression contract pins this invariant:

- `tests/simulation/test_no_cyclic_imports.py` -- spawns a fresh Python
  interpreter for each of the three affected modules (and a combined
  one-process run) and asserts each imports cleanly with no
  `RecursionError` / `ImportError`. Catches the dynamic-import failure
  mode where a top-level statement reintroduces a partial-module re-entry.
- `tests/simulation/test_no_import_cycle.py::test_no_runtime_import_cycles`
  -- builds the runtime import graph (excluding `TYPE_CHECKING` and
  inside-function edges) via `networkx.simple_cycles` and asserts it is
  acyclic. Catches the static-graph failure mode where someone hoists a
  `TYPE_CHECKING` import to module scope and re-creates the cycle.
- `tests/simulation/test_no_import_cycle.py::test_base_does_not_lazy_import_policy_runner`
  -- counts module-level imports of `policy_runner` from `base.py` and
  asserts the count stays at 1 (the documented module-level edge), so a
  re-introduced inline lazy import inside a `SimEngine` method (the
  prior bug shape) fails loudly.

If the runtime cycle is ever reintroduced, those pins fail loudly in CI
-- independent of CodeQL's view of the AST.

**Why a global suppression rather than path-scoped:** CodeQL's
`query-filters` keys filter by `id` / `tags` / `precision` only;
path-scoped exclusion of a single query is not supported (the only
path-aware key is `paths-ignore`, which excludes a file from all
queries -- too broad). The simulation triple is the only file shape in
the repository where this query fires today, so a global exclude is
equivalent in effect *today* and keeps the config small. The cost we
accept is that a future legitimate violation in unrelated code would
be silently suppressed too. Mitigation: the regression pins above
guard against runtime cycles independent of CodeQL, and a future
contributor who introduces a new legitimate violation should drop this
suppression and fix the new cycle properly rather than extend the
filter.

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
