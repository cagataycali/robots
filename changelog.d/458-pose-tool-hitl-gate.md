### Fixed: `pose_tool` asks an operator before a goal position reaches the arm

The five `pose_tool` actions that move the arm (`move_motor`, `move_multiple`,
`incremental_move`, `load_pose`, `reset_to_home`) reached the motor controller's
writers with no gate of their own; the dashboard's `MotionInterruptHook` lists
them, but it is a hook an `Agent` has to be built with, and the canonical
`Agent(tools=robot.tools)` build has none, so the default wiring moved the arm
on any tool call the model emitted (F-010, CWE-862; same class as F-009). Each
now stops for an operator interrupt through the shared command gate BEFORE the
`MotorController` is built; a declined or headless call opens no port.
`STRANDS_POSE_COMMAND_ALLOW` (action names or `*`) pre-approves,
`BYPASS_TOOL_CONSENT=true` lifts the gate with a warning, and a grant the
dashboard hook already deposited is spent instead of asking twice. `connect`,
the reads, the pose library actions and `emergency_stop` are never gated.
