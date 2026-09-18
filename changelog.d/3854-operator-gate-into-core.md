### Changed: the operator-approval gate is a core module, not a private helper of the tools package

`gate_command`, `gate_motion` and the command blocklist move from
`strands_robots/tools/_command_gate.py` to `strands_robots/_command_gate.py`, and
the audit row they write through moves from
`strands_robots/tools/_hitl_audit.py` to `strands_robots/_hitl_audit.py` with
them. Both are private, so no documented import changes. Their callers are six
tool modules and `strands_robots.hardware_robot`, so the layer they belong to is
the lowest of those callers rather than the package that first needed them; the
audit module moves too because the gate reads it at module scope.

That removes the last `app -> tools` inversion declared in
`scripts/check_import_layers.py`, 8 upward runtime edges to 7, and
`tests/test_import_layers_are_a_dag.py` gains a cell pinning what earns the
placement: neither module reads anything above `core` at import time, and the
audit row's one upward read - the mesh safety log it writes through - is deferred
to the call and pinned as exactly that. The single-owner pin on the blocklist now
scans the whole package instead of the tools directory.
