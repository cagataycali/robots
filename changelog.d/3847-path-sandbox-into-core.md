### Changed: the path sandbox is a core module, not a private helper of the tools package

`validate_save_path` and `resolve_output_path` - the guards every surface that
writes a caller-supplied path runs first - move from
`strands_robots/tools/_path_validation.py` to
`strands_robots/_path_validation.py`. Both are private, so no documented import
changes. Their consumers span layers (`strands_robots.training._validate` and
three tool modules) and the module imports nothing but the standard library, so
the layer it belongs to is the lowest of them rather than the package that first
needed it.

That removes the last `sim|policies -> tools` inversion declared in
`scripts/check_import_layers.py`, 18 to 17, and
`tests/test_import_layers_are_a_dag.py` gains a cell pinning what earns the
placement: a core guard reads nothing from the package, in any of the three
import kinds.
