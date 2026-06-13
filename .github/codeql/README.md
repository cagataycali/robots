# CodeQL configuration

This directory holds the CodeQL configuration for `strands-labs/robots`.

## What `config.yml` does

It carries exactly one query suppression:

```yaml
query-filters:
  - exclude:
      id: py/unsafe-cyclic-import
```

## Why `py/unsafe-cyclic-import` is suppressed

The rule fires on the **simulation triple**:

- `strands_robots/simulation/base.py`
- `strands_robots/simulation/policy_runner.py`
- `strands_robots/simulation/benchmark.py`

The flagged cycle is a **compile-time artifact, not a runtime cycle**:

1. Every module in the triple uses `from __future__ import annotations`, so all
   type hints are strings at runtime and are never resolved at import time.
2. `policy_runner.py` imports `SimEngine` from `base` only under
   `TYPE_CHECKING`, so that edge does not exist at runtime either.
3. `base.py` imports `PolicyRunner` / `VideoConfig` from `policy_runner` at
   module level (the R4 form), which is a single one-directional runtime edge —
   no loop is closed.

CodeQL's `py/unsafe-cyclic-import` walks `TYPE_CHECKING` blocks as if they were
runtime edges, so it reports the AST-visible loop. This is the textbook profile
of a documented false-positive.

Runtime safety is pinned by `tests/simulation/test_no_import_cycle.py`, which
builds the module import graph (excluding `TYPE_CHECKING` and in-function
imports) and asserts zero cycles.

## Why repository-wide (not path-scoped)

CodeQL `query-filters` **do not support path scoping**. The exclude is therefore
repository-wide. To keep it honest, two guards make sure it never silently grows
to cover a *legitimate* new cycle:

1. **Schema pin** — `tests/test_codeql_config_schema.py` asserts the exact YAML
   shape (`exclude.id == 'py/unsafe-cyclic-import'`). A typo such as `excludes:`
   or a misspelled rule id fails the unit suite at PR time instead of silently
   no-opping the suppression.
2. **Narrowness CI** — the `codeql-suppression-narrowness` job (see
   `.github/workflows/codeql.yml`) runs the CodeQL CLI **with the suppression
   dropped** and asserts the violating file set is exactly the simulation
   triple. If a new file starts firing the rule, CI goes red and the maintainer
   must either fix the new cycle properly (preferred) or extend the suppression
   with a documented rationale.

## Manual audit recipe

To reproduce the narrowness check locally:

```bash
# 1. Drop the query-filters block (do NOT commit this edit):
yq 'del(.query-filters)' .github/codeql/config.yml > /tmp/config-noexclude.yml

# 2. Build a database and analyze with the cyclic-import query only:
codeql database create db --language=python --source-root=.
codeql database analyze db \
  codeql/python-queries:Imports/UnsafeCyclicImport.ql \
  --format=sarif-latest --output=cyclic-import.sarif

# 3. The only files in the SARIF results MUST be the simulation triple:
#    strands_robots/simulation/base.py
#    strands_robots/simulation/policy_runner.py
#    strands_robots/simulation/benchmark.py
```

If the set has expanded, **do not** widen the suppression reflexively — fix the
new cycle. Only extend the suppression with a written rationale appended here.

## Workflow layout

PR/push and weekly scans are owned by a **single** workflow,
`.github/workflows/codeql.yml` (Strategy A from #237). It scans both `python`
(query suite `security-and-quality`) and `actions` in one matrix, so each PR
runs `Analyze (python)` exactly once. The previous `codeql-advanced.yml` was
removed to eliminate the duplicate Python scan.
