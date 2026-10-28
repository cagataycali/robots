### Added: `Policy("zenoh+tls://host:port/<deployment>")` — a lerobot-policy-server behind a TLS (or ws, quic) zenoh link

`Policy("zenoh://host:port/<deployment>")` always built a `tcp/` endpoint. A policy server reached through a zenoh
router with TLS (the public api.agi.diy slots advertise `tls/zenoh.api.agi.diy:7447`) had to be constructed as
`ZenohPolicy(endpoint="tls/...")` by hand, and the plain URL pointed at a port that does not speak tcp. The URL now
carries the link as the scheme suffix (`zenoh+tls://`, `zenoh+ws://`, `zenoh+quic://`, `zenoh+udp://`), `zenoh://`
stays tcp, an unknown link is refused with the forms listed, and `repr()` round-trips the form.
