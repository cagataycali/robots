### Fixed: a MoveIt2 sidecar reply the client cannot read names the peer, not the codec

`MoveIt2InferenceClient.call_endpoint` decoded every reply frame ungraded, so one
wire fault - a peer on the port that is not the msgpack REQ/REP sidecar - reached
the caller as four different reports, none of them naming the host, the port or
the request that came back. The reference sidecar already grades the mirror image
in the request direction, refusing undecodable bytes and a decodable non-map in
one class because either way the peer did not send a request; the reply direction
is now graded the same way, at a `_decode_reply` seam that raises
`ConnectionError` naming the peer, the endpoint, the problem and the frame's
opening bytes, keeping the codec failure as the cause.

Two of the four reports were silent. `"error" in reply` is a membership test, so
a reply of `"ok"` or `[1, 2]` answered it `False` and was returned as the declared
`dict[str, Any]`, failing later inside the policy as
`AttributeError: 'str' object has no attribute 'get'`; a string that happened to
contain `"error"` took the server-error branch and raised `TypeError: string
indices must be integers` from `reply['error']`. `MsgSerializer.from_bytes` is
annotated `-> Any` to match what `unpackb` actually returns - any msgpack value,
not just a map - with the map requirement graded where the peer and endpoint are
known. `ping` still absorbs the refusal and reports `False`, and a readable
`{"error": ...}` reply keeps its `RuntimeError`.
