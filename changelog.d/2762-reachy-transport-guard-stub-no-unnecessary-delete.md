### Fixed

`tests/drivers/test_reachy_transport_guards_are_reachable.py` no longer opens a
CodeQL `py/unnecessary-delete` alert. The `_StubLink.start` no-op accepted the
driver's two keyword callbacks (`on_joints`, `on_imu`) and immediately
`del`-ed them, which trips the scanner because the locals are about to leave
scope anyway. The keyword names must stay to satisfy `ReachyDriver._impl`'s
call site (`link.start(on_joints=..., on_imu=...)`), so the stub now assigns
them to `_` instead, preserving the shape of the invocation and closing the
required-thread merge gate without touching driver code.
