### Fixed: a policy server that accepts the connection and answers nothing is reported, not waited out

`VeraWebsocketClient` and `Cosmos3WebsocketClient` read their server's metadata
handshake and every action chunk with `websockets.sync`'s `recv()`, which has no
deadline of its own. `open_timeout` covers the TCP connect plus the HTTP upgrade
only, so a server whose listener accepted the connection and then went quiet - a
checkpoint still loading onto the GPU, a wedged forward pass - held the calling
thread indefinitely. Each client's actionable "could not reach the server, start
it first" hint is raised from `except OSError`, and a listening server never
produces one, so the one report that says what to do was unreachable; under
`VeraServerRunner`, whose readiness is a TCP port probe, `start()` returns in
precisely that state.

Both clients now take a `read_timeout` (positive finite seconds, default 600,
refused at construction) that bounds every read, and report a read that expires
as `accepted the connection but sent no ... within read_timeout=Ns` - separately
from the absent-server hint. A missed read also discards the connection: the
reply it did not read is still queued on the socket, so the next request would
otherwise be answered with the previous request's chunk, well-formed and computed
for an observation the robot has already moved past. The handshake is held to the
same rule - published before its metadata frame was read, a failed handshake left
a live connection cached behind the refusal it had just raised, and
`get_server_metadata()` answered `{}` for a server it had never spoken to.
