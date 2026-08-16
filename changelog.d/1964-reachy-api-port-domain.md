### Fixed: a Reachy Mini `api_port` that cannot address a port is refused where it is named

`ReachyMiniDriver(api_port=...)` accepted any value and interpolated it verbatim
into both targets the driver builds - the daemon REST URL and the Lite WebSocket
target - so `api_port=99999`, `True`, `2.7` or `None` produced requests for
`http://host:99999/...`, `http://host:True/...` and `ws://host:None/ws/sdk`.

Nothing downstream refused it. `reachy_transport.api` reports every failure as
an `{"error": ...}` result rather than raising, and for an out-of-range port that
result was byte-identical to the one a reachable port produces with the daemon
down, so the two could not be told apart. `connect()` derives the
Wireless-vs-Lite variant from that result with
`not status.get("wireless_version", True)`, so a result carrying only `error`
read as Wireless: an unusable port silently selected the Zenoh link and
`connect()` logged a successful connection.

The port is now validated at construction through the shared
`utils.tcp_port_error` domain, before any driver state is allocated, so it is
refused at the point a caller names it. The fail-safe that treats a genuinely
unreachable daemon as Wireless is unchanged - a daemon that is down is not a
caller mistake.
