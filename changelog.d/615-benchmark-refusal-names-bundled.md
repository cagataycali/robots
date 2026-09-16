### Fixed: `evaluate_benchmark` refusal points at the bundled benchmarks

An unregistered name was refused with `Registered: []` and a pointer to
`register_benchmark_from_file`, even for the benchmarks that ship in the box.
A bundled name now names the one-call remedy (`register_builtin_benchmarks`);
any other name lists the bundled set beside the file/registration verbs.
