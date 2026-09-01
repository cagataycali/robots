### Fixed: a `PolicyServer` told to stop stops serving the clients it has

`PolicyServer.stop()` closed the listening socket and joined the accept loop,
which is only half of a teardown. `websockets.sync.server.Server.shutdown()`
closes that socket and nothing else, and the sync server keeps no record of the
connections it accepted -- its own source says so where it spawns them:
"since there isn't a mechanism for tracking connections and waiting for them to
terminate". Each connection is served on a thread that outlives the server
object, so `stop()` returned while a client that was already connected went on
streaming observations in and receiving action chunks back. Measured on
websockets 16.1.1: `stop()` returned in 0.18ms and the same open connection was
answered with actions 19 more times over the following second. On a robot that
is the policy still driving the arm after the operator was told the server
stopped. Returning from the foreground `serve()` had the same gap.

`stop()` now closes every client connection still open, and `serve()` does the
same on its way out, so the wrapped policy stops being invoked through either
door. A handler inside an inference call cannot notice the close until that call
returns, so the wait is bounded by `CONNECTION_DRAIN_S` (5s) and the outcome is
reported rather than waited out: a connection still being served after it is
named in a warning, which -- `stop()` returning `None` -- is the only record
there can be of it. The connections are tracked under their own lock, not the
inference lock, so a teardown never queues behind an inference call to find out
what it has to close, and each peer address is recorded when the connection is
accepted, because `remote_address` reads `getpeername()` and raises
`OSError: [Errno 9] Bad file descriptor` once the connection is closed.

Nothing caught it because the lifecycle tests graded the server's own state:
`test_stop_is_idempotent` and `test_context_manager_starts_and_stops` assert
`_server is None`, which is equally true of a server that is still serving.
