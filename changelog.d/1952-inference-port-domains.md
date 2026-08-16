### Fixed: `PolicyServer` and `RemotePolicy` refuse a port they cannot use

The remote-inference pair stored whatever `port` it was handed. The server's went
straight onto `.port`; the client's went verbatim into `f"ws://{host}:{port}"`, so
`ws://127.0.0.1:nan` and `ws://127.0.0.1:[8765]` were constructed and kept. A
WebSocket target is only resolved on first use, so neither was refused by the
transport - each surfaced much later as an unreachable server, implicating the
service the caller was trying to reach rather than the port.

Both halves now validate it, on the two domains the two roles actually have. The
client dials, so it takes `tcp_port_error`'s `[1, 65535]` unchanged. The server
binds, so it takes the same range with the floor at `0` - the documented request
for an ephemeral port, which `start()`/`serve()` still read back onto `.port`. The
refusal precedes `create_policy`, so a port that can never bind no longer builds a
policy first, and precedes the client's `uri`, so no unusable endpoint is stored.

The `--port` CLI flag routes through the same bind rule as the constructor it
calls. Its inline `1 <= port <= 65535` range previously refused `--port 0`, the
ephemeral bind `PolicyServer` documents as first-class.
