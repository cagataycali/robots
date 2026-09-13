### Fixed: a passkey ceremony is refused when no `Host` header arrived, rather than read as loopback

Both halves of a WebAuthn expectation in `strands_robots.dashboard.auth` are
derived from the `Host` header, and each door stood in a value when the header
was absent. `_derive_rp_id` read `"localhost"`, which `rp_id_verdict` answers
`"loopback"` for -- the one verdict that outranks both
`STRANDS_DASH_AUTH_RP_ID` and the enrolled-credential set, because a browser on
this machine is the operator. So a request carrying a host nobody enrolled was
refused, while a request carrying no host at all was trusted absolutely.
`_served_origin` read `"localhost:8090"`, an authority and a port nobody
configured; `origin_verdict` then compares the caller's own `Origin` against
that invention, a comparison the caller passes by offering the invented value --
which is the tautology `origin_verdict` exists to refuse, reached through a
default spelled in the source.

`rp_id_verdict` now refuses an empty host (after the pin, so the deployments the
pin exists for are unaffected), `_derive_rp_id` passes what actually arrived,
and `_served_origin` raises the same 400 `_connection_scheme` raises for a
transport that reports no scheme: an expectation is refused rather than guessed,
and the refusal names the reading that was missing.
