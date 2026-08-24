### Fixed: `DeclarativeBenchmark` refuses a `supported_robots` its own evaluation cannot honor

`DeclarativeBenchmark` has two construction paths. `from_dict` refused a
`supported_robots` that was not a list of strings and refused a `default_robot`
outside it; `__init__` stored `list(supported_robots)` raw. Of the nine checks
`from_dict` runs, only `max_steps` was mirrored - and its comment states the
reason for all of them.

So a single robot name spelled without the list, `supported_robots="panda"`, was
accepted: `str` is iterable per character, so the benchmark declared five
one-letter robots. `list_benchmarks()` advertised
`robots=['p', 'a', 'n', 'd', 'a']`, and `evaluate_benchmark(robot_name="panda")`
then returned `status="error"` refusing the benchmark's *own* `default_robot` and
naming those five as the allowed set. `supported_robots=""` failed the other way:
it stored `[]`, this parameter's documented "any robot" spelling, silently
widening a restricted benchmark to every robot.

Both paths now share the `name_list_error` domain - the same way they already
share `max_steps`' count domain - and `__init__` mirrors the `default_robot`
membership check, so a benchmark can no longer declare a robot set that its own
default robot is outside of. The shape is checked before membership: on a bare
string the membership message would report
`'panda' not in ['p', 'a', 'n', 'd', 'a']`, describing the symptom rather than
the mistake. An empty list is still accepted and still means "any robot".
