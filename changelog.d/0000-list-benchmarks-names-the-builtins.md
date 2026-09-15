### Fixed: the bundled benchmark roster is derived, not hand-written

`list_benchmarks` on a fresh engine said only "Use register_benchmark_from_file
to add one", so an agent looking for a baseline had to find
`register_builtin_benchmarks` by scanning the action list. The empty-registry
text now names both and lists the bundled benchmark names.

`describe()` named the built-ins too, from a hand-written sentence that had
already drifted: `go2_strafe_left` and `go2_turn_left` shipped after it was
written and were never added, so the first surface an agent reads advertised
three of the five. Both texts now read `builtin_benchmark_specs()`, and
`describe()` names the robot each benchmark defaults to.
