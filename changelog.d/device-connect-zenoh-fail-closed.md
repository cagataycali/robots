### Fixed: Device Connect refuses to come online on a transport nobody authenticates

`Robot(...).run()` came online in plaintext by default. `device_connect_edge`
validates transport security only for NATS - its
`DeviceRuntime._validate_startup_config` returns before every check for any
other backend, and its Zenoh adapter never reads `allow_insecure` - so on the
default `zenoh` backend a fresh install with no credentials was reachable on the
LAN unencrypted and unauthenticated, while the code believed it was secure and
the INSECURE warning stayed silent. `docs/device-connect.md` promised "Secure by
default" for exactly that path.

`init_device_connect` now resolves whether anything will authenticate the
transport - a credentials or TLS file variable, or a `tls` / `quic` /
`zenoh+tls` / `mqtts` endpoint, with NATS still left to the edge's own check -
and refuses to build the runtime when nothing will. The refusal names both
remedies: point the device at credentials, or opt in for a trusted isolated
network with `DEVICE_CONNECT_ALLOW_INSECURE=true` and restrict who may drive it
with `DEVICE_CONNECT_RPC_ALLOW`. The opt-in path is unchanged, and still logs
the INSECURE warning for as long as it is active.
