### Added: `run(policy="hold")` stands still

`"hold"` is now a policy spec: `robot.run("hold", seconds=5)` and `sim.run({"so101": "hold"}, seconds=5)` keep each
body at the pose its run started at, exactly where `robot.hold()` plus stepping the world would leave it. Before, the
word an agent sees as an rpc was refused as `Policy('hold'): unknown form`. The refusal for an unknown preset now lists
`hold` with the ONNX presets.
