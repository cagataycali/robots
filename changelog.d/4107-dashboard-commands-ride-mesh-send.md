### Fixed: a dashboard command is a `Mesh.send`, so a forged reply is audited and an unusable wait budget is refused

`MeshBridge.send_cmd` built its own envelope, published it on the raw session
and correlated the reply in a private `_on_response`, beside the `Mesh` the
bridge already holds for the signed safety rail. That copy dropped a forged
reply with a log line and no `response_hijack_rejected` record in
`mesh_audit.jsonl`, and put a command on the wire with a `nan` or negative
wait (returning a timeout at once) or raised `OverflowError` on `inf`.
`send_cmd` now calls that `Mesh`'s `send()`; the envelope, the reply topic and
the correlation tables are removed. The return contract is unchanged: the
peer's response, or `{"ok": False, "error": ...}`.
