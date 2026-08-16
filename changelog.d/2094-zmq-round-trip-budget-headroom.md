### Quality: a live ZMQ round trip is no longer required inside a 2 ms budget

`tests/test_zmq_timeout_ms_domain.py` asserted `client.ping() is True` for every
usable `timeout_ms`, including a 2 ms one. A fresh REQ socket pays the TCP
connect and the ZMQ handshake on its first call, and that cost is scheduler-bound
rather than transport-bound, so the assertion held on an idle host and failed
under CPU contention - it measured the runner rather than the value reaching the
socket.

The property the file exists to pin - that the coerced budget is what the socket
was configured with - is asserted through `getsockopt` for every usable spelling,
which needs no clock. The live round trip now runs only for budgets with headroom
over the connect cost, a set derived from the usable table rather than written out
beside it, and a structural guard keeps a live answer from being required inside a
budget the scheduler can exceed.
