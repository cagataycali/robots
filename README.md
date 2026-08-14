# newton `add_robot` refusal coverage — measurements

Tests-only change: **0 production lines**. These are the measurements behind the PR.

## Files
- `newton_add_robot_refusal_coverage.png` — the figure (composed by `compose.py`, every
  number read from the JSONs below, nothing typed by hand).
- `covsum.json` — per-refusal-site coverage before/after, from
  `pytest tests/simulation/newton --cov=strands_robots --cov-report=json`, read out of
  `coverage.json -> files[newton/simulation.py].missing_lines`.
- `rows.json` — the mutation table: for each plausible regression, the failure count under the
  new cases and under the pre-existing suite.
- `compose.py` — the figure generator. It asserts every rendered claim against the JSONs
  (all five sites unexecuted on main, none after, no line regressed, all-caught/all-blind),
  plus a per-axes text-placement guard, a derived row pitch and a clean 8px border.

## Method
1. **Site census** — AST-walk `NewtonSimEngine.add_robot` for every `return {"status": "error"...}`
   and check each line against the coverage JSON's `missing_lines`.
2. **Mutation table** — each anchor scoped to `add_robot` by AST line range (asserted unique
   inside the function), applied, both arms run without `-x`, then restored and the source
   asserted byte-identical.
