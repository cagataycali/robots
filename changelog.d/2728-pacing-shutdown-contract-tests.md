### Tests: the pacing shutdown paths that promise never to raise are graded

`Ticker.wake` and `Ticker._drain` document that they never raise, and
`Ticker.close` that it is additionally idempotent. Nothing graded any of those
three promises (#2728): the bodies of the five exception handlers they are made
of were the only uncovered lines left in `strands_robots/mesh/pacing.py`, so the
happy paths were tested and the reason each handler exists was not.

The promises are load-bearing rather than defensive decoration. Every mesh
publish loop paces on a `Ticker`, and `Ticker.wait` sits *outside* the `try` in
each of those loops - the same placement the module docstring explains for the
Windows doorbell. A shutdown path that raises therefore does not surface as a
handled error: it kills the publish thread while the mesh itself still looks up,
which is the module's own description of the hardest shape of this failure to
attribute, "a robot that joins the fleet and streams nothing".

`TestShutdownKeepsItsPromisesWhenADescriptorDiesUnderIt` drives each handler
through the condition it exists for - the descriptor dying *under* the ticker,
which is what a concurrent close or a reaped fd looks like from in there - and
asserts a post-condition rather than only that nothing was raised. A call that
merely returns would also pass against a handler that swallowed the error and
abandoned the rest of the shutdown, so `close` is checked to have completed (a
subsequent `wait` reports the ticker as closed) and a failed `wake` is checked
not to have cost the caller the stop itself.

Three of the five conditions are reached with real descriptors: a reaped read
end, a closed write end, and a send buffer filled until it would block, whose
`BlockingIOError` lands in the same handler as a closed socket. The other two
drive a stand-in, for the reason `TestTheDoorbellIsSomethingEverySelectorAccepts`
already states about the Windows selector - a POSIX `epoll` releases its fd
idempotently and `socket.close()` does not raise on a double close, so on this
host those two handlers cannot be reached with the real objects. They guard a
platform that does raise, so the stand-in pins the contract instead of the
platform.

The socket case asserts that *both* halves of the doorbell were released even
though the first raised. That is the assertion the handler is really for: a
shutdown that gave up at the first failing `close()` would still never raise,
and would leak a descriptor per paced loop.

This takes `strands_robots/mesh/pacing.py` from 89% to 100% statement coverage -
10 uncovered statements to 0 - with no production change.
