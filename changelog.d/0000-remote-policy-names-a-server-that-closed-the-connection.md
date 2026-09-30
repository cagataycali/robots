### Fixed: a policy server that closes the connection is reported by endpoint

When a `PolicyServer` was stopped or crashed during a rollout, `run_policy`
reported only websockets' own `received 1001 (going away); then sent 1001
(going away)`. `RemotePolicy` now reports that the server at that URI closed
the connection before sending its reply or handshake. The report quotes the
close code and reason and says the next call dials again.
