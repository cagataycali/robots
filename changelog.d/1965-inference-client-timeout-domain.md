### Fixed: an inference-client `timeout_ms` that cannot be a wait budget is refused

`Gr00tInferenceClient`, `MoveIt2InferenceClient` and the `MoveIt2Policy` that
forwards to one write a caller-supplied `timeout_ms` straight into ZMQ's
`RCVTIMEO`/`SNDTIMEO`, and none of them checked it. Measured against a sidecar
answering in 5 ms, `timeout_ms=0` and `timeout_ms=True` were accepted and then
made every request fail with `zmq.Again` in under a millisecond - a healthy
service reported unreachable by the caller's own configuration - while `-1` was
accepted as ZMQ's infinite receive, the unbounded block the `LINGER, 0` two
lines below the same `setsockopt` exists to remove. `-5000`, `2.7`, `nan`,
`inf`, `"15000"`, `None` and a list each escaped the constructor as a bare
`ZMQError`/`TypeError` from pyzmq.

All three now validate it with the shared `positive_count_error` domain before
any socket is created, so a refused budget dials nothing and reports the same
message wherever it is supplied. The socket option is a C `int`, so an integral
float such as `15000.0` is refused by `setsockopt` rather than coerced - the
same reason that domain already covers `range()` bounds and framebuffer
dimensions. An AST guard keeps a fourth sidecar client from taking `timeout_ms`
without it. Accepted budgets, including the `15000` default, are unchanged.
