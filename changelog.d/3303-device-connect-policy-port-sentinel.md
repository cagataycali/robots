### Bug Fixes

- **device_connect**: the `execute` RPC read `policy_port` with `or`, so every
  falsy spelling the wire can carry - not just the documented integer-`0`
  sentinel - reached `start_task` as `None`. A supplied-but-unusable `0.0`,
  `False`, `""` or `[]` was therefore reported as `"policy_port is required to
  build a policy"` by a port-dialing provider, and silently dropped (task
  started) by a port-less one, while the truthy malformed spellings were refused
  by name. Only the integer `0` now maps onto "not supplied"; every other value
  is handed to `Robot._policy_port_error`, the one surface that holds the port
  to the shared `tcp_port_error` domain and names the value the caller supplied.
