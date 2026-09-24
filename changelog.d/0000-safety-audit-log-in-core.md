### Changed: the safety audit log sits in `core`, not in `mesh`

`strands_robots.mesh.audit` is now `strands_robots.audit`. Behaviour, on-disk
format and the four `STRANDS_MESH_AUDIT_*` variables are unchanged: it is the
same file, `~/.strands_robots/mesh_audit.jsonl`, written by the same code.

The module imports nothing from the package - stdlib only - and has writers in
three layers: the mesh peers that name the file, the `robot_mesh` tool, and
`_hitl_audit`, which writes the operator-response row every human-in-the-loop
gate owes. `_hitl_audit` is a `core` module, so under `mesh` it reached two
layers up for a JSONL appender, and the only way to keep that legal was to defer
the import into the call - a deferral whose stated reason was the transport stack
that `strands_robots.mesh` pulls in, not the log itself.

With the log under all three writers the deferral goes with it:

```
upward deferred edges: 4 -> 3      (core -> drivers|mesh group: 1 -> 0)
```

`_hitl_audit` now binds the module rather than the function, so
`log_safety_event` is resolved on the sink at call time: one patch target for
every gate's rows, which is what lets a test assert that all four gates reach
the same log.

`tests/test_import_layers_are_a_dag.py` grades the new placement the way it
grades the dataset contracts - layer, the reads that allow it, and the caller
layers that need it - and `_hitl_audit`'s allowed-deferral set is now empty, so
a second upward dependency cannot join it unnoticed.
