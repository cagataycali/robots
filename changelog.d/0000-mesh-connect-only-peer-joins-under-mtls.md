### Fixed: a mesh peer that only sets `ZENOH_CONNECT` joins under the default mTLS posture

A peer with `ZENOH_CONNECT` and no `ZENOH_LISTEN` - the documented way for a
second host to join - never opened its session under `STRANDS_MESH_AUTH_MODE=mtls`:
Zenoh still opens a listener for a dialling peer, its default is `tcp/[::]:0`, and
the mTLS posture restricts the transport to TLS, so the open failed with
`Unsupported protocol: tcp` and the peer stayed off the mesh. A connect-only peer
now listens on the same default in the posture's scheme (`tls/[::]:0` under mTLS,
`tcp/[::]:0` otherwise). `tests_integ/acceptance/` gains the two-peer check: two
processes, an explicit `tls/` endpoint, state, a `status` RPC, and an e-stop that
only the right resume code clears.
