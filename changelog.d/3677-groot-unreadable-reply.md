### Fixed: a GR00T policy-server reply the client cannot read names the peer, not the codec

`Gr00tInferenceClient.call_endpoint` decoded every reply frame ungraded, so one
wire fault - a peer on the port that is not the msgpack REQ/REP GR00T policy
server - reached the caller as three different reports, none of them naming the
host, the port or the request that came back. The reference client already
refuses one such frame by name, the legacy `b"ERROR"` sentinel, and reads the
`error` field only off a reply that is a map; the reply direction is now graded
in that spirit at a `_decode_reply` seam that raises `ConnectionError` naming
the peer, the endpoint, the problem and the frame's opening bytes, keeping the
codec failure as the cause.

Two of the three reports were silent. `"error" in response` is a membership
test, so a reply of `"ok"` answered it `False` and was returned as the declared
`dict`, failing later inside the policy as `AttributeError: 'str' object has no
attribute 'items'`; a string that happened to contain `"error"` took the
server-error branch and raised `TypeError: string indices must be integers`.
Unlike the MoveIt2 sidecar, the reference server sends a list on purpose -
`get_action` returns `(action, info)`, which msgpack carries as a 2-element list
- so the seam admits a map or a list, and `get_action` grades that envelope
itself rather than returning whatever list arrived as the action dict: `[1, 2]`
used to return `1`. `MsgSerializer.from_bytes` is annotated `-> Any` to match
what `unpackb` actually returns. `ping` still absorbs the refusal and reports
`False`, `Gr00tPolicy.reset` still logs it and continues, and a readable
`{"error": ...}` reply keeps its `RuntimeError`.
