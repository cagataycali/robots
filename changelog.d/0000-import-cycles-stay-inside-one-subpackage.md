### Fixed: the policies <-> simulation <-> registry <-> drivers import cycle

Twenty-one modules across four subpackages formed one import cycle, held
together by function-local imports: `drivers/base.py`, `registry/policies.py`
and `simulation/base.py` imported `Policy` from the `policies` package instead
of `policies.base`, and `policies/_rng.py` reached the seed domain through
`simulation.base`, which imports `policy_runner`, which imports `policies`.
`MAX_EVAL_SEED` and `randomization_seed_error` now live in the leaf
`strands_robots.simulation._seed` (re-exported from `simulation.base`, so
callers and monkeypatch sites are unchanged), and the three `Policy` imports
target `policies.base`. A new grader,
`tests/test_import_cycles_stay_inside_one_subpackage.py`, walks the whole
package by AST and refuses any cycle that crosses a subpackage boundary.
