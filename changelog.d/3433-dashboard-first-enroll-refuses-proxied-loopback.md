### Fixed: the dashboard's first passkey enrollment refuses a loopback peer that arrived through a proxy

`begin_registration` admits the ownership-granting first enrollment from the
machine itself when no `STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN` is set, deciding
"the machine" from the socket peer. Behind the same-host `cloudflared` tunnel
the docs describe, started without `--proxy-headers`, every remote visitor's
peer is `127.0.0.1`, so a stranger who reached the tunnel during the
pre-enrollment window could enroll the owner passkey (F-007, CWE-290 /
CWE-348). A loopback request that carries any proxy forwarding header
(`x-forwarded-for`, `x-forwarded-proto`, `x-forwarded-host`, `x-real-ip`,
`cf-connecting-ip`, `cf-ray`, `forwarded`) is now refused with a message naming
the header, the bootstrap token and the `--proxy-headers` /
`--forwarded-allow-ips` remedy; header values are never read, only their
presence. A browser on the machine sends none of them and is still let in; the
bootstrap token still works from anywhere. `docs/dashboard/remote-access.md`
tells operators to enroll locally or set the token before `cloudflared tunnel
run`.
