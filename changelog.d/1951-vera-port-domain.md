### Fixed: the VERA policy ports take the shared TCP-port domain

`VeraConfig` stored `server_port` / `vis_port` verbatim, and three consumers then
read the field under three different coercions: the provider dialed
`int(server_port or 0)`, `VeraConfig.server_uri` interpolated it verbatim, and
the server runner's argv carried `str(server_port)`. An unusable value was not
merely refused late but *applied* as three different ports - `server_port=2.7`
dialed `ws://host:2` while launching `--port 2.7` and reporting
`ws://host:2.7`, so the client could not reach the server it had just started,
and `True` produced the non-URI `ws://host:True`. `0`, `-1` and `70000` were
accepted the same way; `nan`, `inf` and a list escaped the constructor as a bare
`ValueError` / `OverflowError` / `TypeError` naming neither the field nor the
class.

Both ports are now checked once, on the effective value, in
`VeraConfig.__post_init__` - the one funnel the provider keywords, a pre-built
config and the `VERA_*_PORT` environment overrides all pass through - against
`tcp_port_error`, the same domain the other port-dialing providers apply.
`vis_port = 0` keeps its documented meaning (disable the live viewer) via a
wrapper that decides only that floor and defers the range, so the two ports
cannot drift apart on what counts as an addressable port. The refusal precedes
the client and the runner, so a rejected port leaves nothing half-configured.
