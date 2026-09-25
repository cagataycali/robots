### Fixed: a loopback WebSocket a test opened is closed when that test ends, and `Cosmos3WebsocketClient.close()` exists to do it

`tests/policies/test_policy_client_wire_reads_state_a_deadline.py` opened six
loopback listeners and two client connections per run and closed none of them.
A `websockets.sync` connection is not idle while it is open: each side runs a
keepalive thread that draws `random.getrandbits(32)` for every ping it sends,
from the process-global Python `random`, for as long as the connection lives -
so on an xdist worker that had run this module, four threads went on perturbing
the stream every test after it read. `tests/policies/test_rng_parity.py`
compares two `reset(seed=4242)` windows of Python and NumPy draws, and a ping
landing inside one window read as the Python stream replaying differently while
NumPy's replayed the same, which is the signature #4023 records against `main`.
The module's listeners and clients now belong to a `loopback` fixture that
releases the parked handlers, closes every client, shuts every server down and
refuses a thread that outlives the test, so the leak is reported at the test
that made it rather than at whichever later test happened to read the stream.

`Cosmos3WebsocketClient` had no way to let its connection go: the raw transport
underneath it had a `close()`, the client did not, so a caller holding one kept
its keepalive thread until the process exited. `close()` drops the connection
if one is open, is idempotent, dials nothing to find out there is nothing to
drop, and does not retire the client - the next `infer` connects again, the
same lazy dial the first call performed. Closes #4023.
