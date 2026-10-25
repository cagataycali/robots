### Changed: `act(action, seconds=)` reaches a pose; `Policy("hold")` is gone

`robot.act(action, seconds=s)` ramps the targets from the measured pose to `action` at 30 Hz, settles there and reads
back the pose reached; an agent tool's `act` (`confirm_act`) asks the operator gate once, for the target. One
plain `act` of a far target was read back mid-swing, so an agent reached for `run("hold")` to "move for longer": a run
that latched the first pose and moved nothing, approved by the operator anyway. `Policy("hold")` is now refused with
`use robot.hold() to stand still, robot.act(..., seconds=) to reach a pose`, and `hold` is no longer listed as a
preset. The device verb is unchanged: the `hold` rpc (`robot.hold()`, the mesh `hold`, `broadcast("hold")`) and
`on_disconnect="hold"` still hold at the measured pose.
