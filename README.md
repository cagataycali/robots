# Artifact: a live ZMQ round trip inside a 2 ms budget measures the runner

`capture.py <tree> <tag>` measures, per tree, the fresh-socket connect+handshake
cost and the verdict of `tests/test_zmq_timeout_ms_domain.py` idle and under CPU
contention. `compose.py` builds the figure and asserts every rendered number
against the two dumps, including that the two arms measured different trees.
