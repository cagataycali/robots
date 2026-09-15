### Fixed: a frame `RemotePolicy` cannot read names the peer instead of the codec

Every other malformation a peer can send on the remote-inference WebSocket is
already a `ConnectionError` naming the URI - a first frame that is not `ready`,
a protocol version this client does not speak, a metadata field outside its
domain. A frame the *codec* could not read was the exception:
`protocol.loads` raises a `ValueError`
(`UnicodeDecodeError: 'utf-8' codec can't decode byte 0x82 in position 0` for
non-UTF-8 bytes, `JSONDecodeError` for text that is not JSON), and it escaped
from `_connect` and `_request` - both documented to raise `ConnectionError` -
naming neither the endpoint nor which read it answered.

That is the report for an ordinary mistake, because this package also serves
policies over a WebSocket in msgpack (`policies.cosmos3`): dialling that server,
or any other endpoint on the host, answered `invalid start byte`. Both reads now
decode through one seam that reports the URI, whether it was the handshake or a
reply, the codec failure (kept as the exception's cause) and the frame's opening
bytes, so an operator can recognise what they actually reached. The unfinished
exchange is still discarded, so a retry opens a fresh connection.

This is the client half of a rule the protocol already had: `PolicyServer`
marshals a frame it cannot parse back as an `error` message and carries on
serving the same connection.
