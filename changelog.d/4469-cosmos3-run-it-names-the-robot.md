### Fixed: the Cosmos 3 "Run it" rollout reaches the policy server

The fence on `docs/learn/policies/cosmos3.md` now passes `robot="franka"`, so
`run_policy` dials the server instead of stopping at the preflight that refuses
actions no actuator would receive.
