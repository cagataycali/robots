### drivers/microduck

`MicroduckDriver.send_action({"skill": X})` and the sibling `do()` helper
now add a `Did you mean` hint to the `unknown skill` refusal, mirroring the
pattern used in `hardware_robot`, `robot`, `policies.factory` and
`teleoperator`. When the typo matches an ONNX actor shipped under
`strands_robots.policies.microduck` (e.g. `alpha_walking`,
`ball_kick_left`), the message also names the right surface
(`create_policy("microduck", ...)` / `sim.run_policy(policy_provider="microduck", ...)`),
turning what used to be a stone wall between the two "skill" vocabularies
into a one-line nudge. `kick_left` is suggested for `ball_kick_left`,
`ground_pick` for `alpha_ground_pick`, and so on. The refusal still begins
with `unknown skill` so the pin test
(`tests/drivers/microduck/test_microduck_driver_over_socket.py`) is
preserved.
