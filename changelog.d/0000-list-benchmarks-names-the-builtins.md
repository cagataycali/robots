### Fixed: an empty `list_benchmarks` names `register_builtin_benchmarks`

`list_benchmarks` on a fresh engine said only "Use register_benchmark_from_file
to add one", so an agent looking for a baseline had to find
`register_builtin_benchmarks` by scanning the action list. The empty-registry
text now names both and lists the bundled benchmark names (read from
`builtin_benchmark_specs()`, so it cannot drift).
