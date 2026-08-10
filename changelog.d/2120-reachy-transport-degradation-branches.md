### Quality: drive the Reachy transport's three degradation branches

`strands_robots.device_connect.reachy_transport` carries three tolerance branches
that keep a hardware link usable when the environment is imperfect, and none of
them was exercised. `ZenohLink.start` registers two byte-identical wrapper
callbacks and its docstring states the drop-a-malformed-frame contract for both
topics, but only the joints wrapper had ever been driven with a bad frame.
`resolve_host` passes an unresolvable hostname through unchanged, with only its
resolve path pinned. `WebSocketLink.start` falls back to the legacy
`extra_headers` keyword when `websockets.connect` cannot be introspected -- the
path that carries the bearer credential, so a change dropping the headers there
would have connected unauthenticated with the whole suite green.

Six tests now cover all three, a dropped frame is shown to leave the
subscription still delivering, and the two production docstrings that omitted
the fallback they carry now describe it.
