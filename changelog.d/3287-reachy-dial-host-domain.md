### Fixed: the host half of the Reachy Mini daemon address is graded beside the port

`ReachyMiniDriver` grades the two other values its constructor takes -- `api_port`
through the shared `tcp_port_error` domain and `prefix` through the key-prefix
rule -- and stored `host` unchecked. All three are interpolated into targets the
module builds itself: `http://<host>:<api_port>` in `reachy_transport.api` and
`ws://<host>:<api_port>/ws/sdk` in `WebSocketLink`.

The host half is what can discard the port half, so grading the port alone was
not enough. `host="127.0.0.1/foo"` builds `http://127.0.0.1/foo:8000/...`, which
resolves as host `127.0.0.1` with the validated port sitting in the *path*, so
the driver dials `:80` -- a port nobody configured, and one the port domain
cannot see, because it is the host that discards it. `"bot.local@evil.example"`
resolves to `evil.example` on the configured port. A non-string is carried
verbatim, so `None` reaches the resolver as the DNS name `"none"` and the device
identity reports `Reachy Mini @ None`.

None of these were refused downstream. `api` reports every failure as an
`{"error": ...}` result rather than raising, so an unusable host surfaced as an
unreachable daemon -- the same report a reachable host produces with the daemon
down -- and `connect()` then read that result as the Wireless variant and logged
a successful connection over Zenoh.

`host` now takes the shared `dial_host_error` domain, beside the port it is
dialled with, so it is refused where the caller names it and before any driver
state is allocated. Nothing that addressed a Mini changes: `reachy-mini.local`,
an IPv4 literal, `localhost`, `0.0.0.0` and `[::1]` are all still accepted. The
documented fail-safe that treats an unreachable daemon as Wireless is unchanged
-- a daemon that is down is not a caller mistake.
