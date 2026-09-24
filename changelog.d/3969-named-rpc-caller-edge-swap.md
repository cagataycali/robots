### Fixed: the named RPC caller survives the edge swap a test performs

`named_rpc_caller` (tests/conftest.py) runs a test as an allowlisted Device
Connect operator by binding a stub over `get_rpc_source_device`. It can only
patch what is registered when it runs, and a `MagicMock` stand-in is not
patchable -- it carries no `__file__`, so the fixture skips it and the mock
answers for itself. A test that then called
`tests._device_connect_real.use_the_real_edge` took that mock away and
re-imported the integration against the real edge, which reads the wire's
`source_device`: the allowlist still held the fixture's operator, so the RPC was
refused before it reached the robot.

Collection precedes every teardown, so a stand-in installed by one of the two
files that install one is resident for every file collected after it, and the
first cell to reach the swap paid for it. Measured with the whole tree collected
and one test selected, the first of four cells in
`tests/test_hardware_policy_port_domain.py` reported `caller not authorized for
'execute'` with `caller='op-1'` while the three behind it passed on the edge that
cell had just made real -- so a `-k` filtered run reported a failure a run of the
same file did not.

The stub is registered in `tests._device_connect_real.EDGE_REBINDERS` as well,
and the swap re-applies it to the edge modules it imports, so the integration
binds the caller the session named. The registration is a
`monkeypatch.setitem`, so it does not outlive the fixture.
