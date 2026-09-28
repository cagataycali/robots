### Fixed: the CodeQL alerts in `strands_robots/simulation`, starting with the base/policy_runner/benchmark import cycle

`simulation.base` no longer imports `simulation.policy_runner` at module level,
which was the one import-time edge that closed CodeQL's cycle through `base`,
`policy_runner` and `benchmark` (five `py/unsafe-cyclic-import` errors: `SimEngine`,
`PolicyRunner`, `VideoConfig` and `BenchmarkProtocol` undefined depending on import
order). `VideoConfig` now lives in `simulation.video_config`, below both modules;
importing it from `policy_runner` keeps working. `PolicyRunner` is imported inside
the methods that construct one. Log lines in `model_registry` no longer interpolate
caller-supplied values; empty `except` clauses in the MuJoCo and Isaac backends say
why they are empty. No behaviour change.
