### Fixed: a Cosmos 3 frame the client cannot read names the peer, not the codec

`Cosmos3WebsocketClient` reports every other failure on its msgpack WebSocket as
a `ConnectionError` naming the server and the endpoint: a refused connect
carries the start-the-server hint, a read that expired carries the
silent-server report and the budget it waited out. A frame the *codec* could not
read was the one that did not - `msgpack` raises `ExtraData` (a `ValueError`)
for anything that is not a msgpack document and a `TypeError` for a text frame,
and both escaped `get_server_metadata` and `infer` as `unpack(b) received extra
data.`, naming neither the server, the endpoint, nor which read the peer had
answered. Dialling a proxy that returns a 502 page, a JSON policy server, or
this package's own `inference.server` on the wrong port all reported the same
way. Both doors now report the service, the URI, the read (`metadata handshake`
/ `reply`, the words the silent-server report already uses) and the frame's
opening bytes, keeping the codec failure as the cause; the reply door's
server-error-string contract and the handshake's connection discard are
unchanged. This is the client half of a rule this package already applies at the
other end of its own wires, where `inference.server` marshals an unreadable
frame back as an `error` message and the MoveIt 2 node replies
`malformed_request`. Pinned by
`tests/policies/cosmos3/test_websocket_client_transport.py`.
