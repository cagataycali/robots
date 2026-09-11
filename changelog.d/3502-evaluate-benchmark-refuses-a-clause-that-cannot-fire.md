### Fixed: a benchmark clause that can never fire is refused, not scored 0%

`evaluate_benchmark` now resolves the entities a spec's `success` / `failure` /
`dense_reward` clauses name against the live scene before it builds a policy, the
probe `run_policy` already ran over a `stop_when` clause authored in the same DSL.
A name the scene does not answer to made its term a constant, so a success clause
became unsatisfiable and the eval reported `success_rate: 0.0` beside
`success_measured: true` under `status="success"` - one character apart from `1.0`,
and indistinguishable from an honest policy failure. A `dense_reward` term over a
missing body went dead at `0.0` the same way. Benchmarks a pre-eval probe cannot
decide are evaluated unchanged: one declaring its own `scene` (loaded after the
probe would run) and one constructed from already-compiled clause callables.
`DeclarativeBenchmark.referenced_entities()` exposes what a spec references, so a
Python benchmark can opt into the same guard.
