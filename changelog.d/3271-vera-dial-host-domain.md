### Fixed: the VERA policy host is the other half of the server URI, and takes a domain

`VeraConfig.server_uri` is `ws://{host}:{port}`. The port half was held to the
shared TCP-port domain; the host half was held to nothing, so a value that is not
a host was resolved rather than refused. `host="127.0.0.1/foo"` and
`host="ws://127.0.0.1"` re-cut the URI so the client dialed port **80**,
discarding the validated port. `host=""` was the one unusable spelling the
readiness probe accepted — it maps a bind-only host to loopback — so the runner
reported `VERA server ready` and the client then raised `InvalidURI` past the
`OSError` channel carrying its actionable hint. A non-string raised a
`getaddrinfo` `TypeError` out of `start()`, past the runner's documented error
channel.

`host` is now checked in the same funnel the port passes through: a bare hostname
or IP literal, with `[::1]` as the bracketed IPv6 spelling and `"0.0.0.0"` named
in the refusal for `""` as the way to reach a server bound on every interface.
Whether the host resolves is left to the readiness probe, the surface that can
observe it.
