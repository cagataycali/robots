# CodeQL configuration

This directory contains the CodeQL configuration consumed by both
`.github/workflows/codeql.yml` and `.github/workflows/codeql-advanced.yml`
via the `config-file:` key on the `github/codeql-action/init` step.

## Files

- `config.yml` -- query filter rules. See inline comments for per-rule
  rationale.

## Query suite vs config-file interaction

The `queries:` key on `github/codeql-action/init` (set to
`security-and-quality` in `codeql.yml`, default in `codeql-advanced.yml`)
selects which CodeQL query *suite* runs. The `config-file:` value points
at this directory's `config.yml`, which defines `query-filters`.

Per the [CodeQL Action docs][codeql-config-docs] and the comment that
already lives at `.github/workflows/codeql-advanced.yml:78-79`:

- A workflow-level `queries:` value *replaces* a config-file-level
  `queries:` value unless the workflow value is prefixed with `+`.
- `query-filters` from the config-file always layer *on top* of whichever
  suite is active.

`config.yml` here intentionally does not define a `queries:` key, so the
workflow-selected suite (`security-and-quality` in `codeql.yml`) remains
the active set; this config only contributes the `query-filters` layer.
If a future contributor adds `queries:` to `config.yml`, the suite will
*replace* the workflow's `security-and-quality` selection unless
explicitly merged with `+queries: ...` in the workflow.

[codeql-config-docs]: https://docs.github.com/en/code-security/code-scanning/automatically-scanning-your-code-for-vulnerabilities-and-errors/customizing-your-advanced-setup-for-code-scanning#using-a-custom-configuration-file

## Suppressed queries

### `py/unsafe-cyclic-import` (global)

**Why it fires:** the rule walks `if TYPE_CHECKING:` blocks for cycle
detection. `strands_robots/simulation/{base,policy_runner,benchmark}.py`
form a deliberate static-only cycle so `policy_runner` can call into the
`SimEngine` ABC defined in `base`, while `base` can advertise
`PolicyRunner`-typed return values to static checkers (mypy, IDE
navigation, `typing.get_type_hints` consumers).

**Why it is safe at runtime:** the cycle is asymmetric, not closed.

- `simulation/base.py`'s top-level `from
  strands_robots.simulation.policy_runner import PolicyRunner,
  VideoConfig` is the runtime edge that static analysers see and flag.
  (Cited by symbol rather than line number so future edits above the
  import don't silently invalidate this rationale.)
- `simulation/policy_runner.py`'s `if TYPE_CHECKING:` block imports
  `SimEngine`, `BenchmarkProtocol`, and `Policy`. These are not real
  edges at import time -- `TYPE_CHECKING` is `False` at runtime, the
  block is skipped, and the names exist only for the static type
  system.
- `simulation/benchmark.py`'s `if TYPE_CHECKING:` block similarly
  imports `SimEngine`.

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
  `RecursionError` / `ImportError` traceback frames in stderr (anchored
  on the start-of-line `ExceptionName:` framing rather than a substring
  match, so benign log lines that mention the exception names by name
  do not fail the pin). Catches the dynamic-import failure mode where
  a top-level statement reintroduces a partial-module re-entry.
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

**Periodic audit reminder:** the assumption that the simulation triple
is the only file shape that fires this rule today drifts silently as
the codebase grows. Maintainers should periodically (every minor
release or every six months, whichever is sooner) enumerate what
`py/unsafe-cyclic-import` would flag absent the filter, e.g. by running
the CodeQL CLI locally with the suppression dropped:

```bash
# Drop the exclude block in .github/codeql/config.yml temporarily, then:
codeql database create db --language=python --source-root=.
codeql database analyze db \
    codeql/python-queries:Imports/UnsafeCyclicImport.ql \
    --format=sarif-latest --output=cyclic-import.sarif
# Inspect cyclic-import.sarif: every result outside the simulation
# triple is a candidate true-positive that the global exclude is
# silently hiding.
```

If results outside the simulation triple appear, drop the suppression
(or narrow it via path-scoped surgery) and fix the new cycle properly.

**References:**

- PR #209 -- the multi-round attempt to satisfy CodeQL via code surgery.
  Paused at draft after the constraint conflict between mypy and
  `py/unsafe-cyclic-import` was confirmed locally; full analysis in
  the S13/R6 changelog of that PR.
- Issue #215 -- this follow-up tracking the suppression.
- CodeQL alerts #83, #84 -- closed by PR #209 R4.
- CodeQL alerts #85, #86, #87 -- raised by the rule against the
  `if TYPE_CHECKING:` blocks in `benchmark.py` and `policy_runner.py`;
  closed by this suppression. (Historical line numbers in the alert
  bodies: `benchmark.py:42`, `policy_runner.py:49-50` at the SHA that
  raised the alerts.)
- CodeQL alerts #253, #254, #255 -- recurrent alerts on `base.py`'s
  `if TYPE_CHECKING:` block introduced by PR #209 R5 attempting to
  recover mypy type info; closed by this suppression. (Historical line
  number: `base.py:28` at the SHA that raised the alerts.)

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
