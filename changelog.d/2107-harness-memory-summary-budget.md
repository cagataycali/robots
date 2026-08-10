### Fixed: the harness-memory summary budget now covers the payload the store holds

`HarnessMemory.save_trace` writes the caller's summary plus a provenance block
(timestamp, library version, backend, robot) into one file, and `load_trace`
re-validates that file before anything reaches planner context. The save side
measured the caller's payload while the load side measured the caller's payload
plus provenance, so a summary in the top 140 bytes of the documented 64 KiB
budget saved with `status="success"` and was then permanently unreadable -- and
the remedy the load failure names ("delete it with delete_trace and re-save")
reproduced the same unloadable file. `save_trace` now checks what it is about to
store, before writing, so every summary a save accepts is one a load accepts.
The trace budget was already symmetric and is unchanged.
