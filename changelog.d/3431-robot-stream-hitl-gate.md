### Fixed: the `Robot` agent tool asks an operator before it drives real hardware

`strands_robots.hardware_robot.Robot` - what `Robot("so100", mode="real")`
returns, and the README's first path to metal once handed to an `Agent` -
dispatched `execute` to `_execute_task_sync` and `start` to `start_task` with no
gate: its `stream` never read a tool context, never raised an interrupt and never
consulted `BYPASS_TOOL_CONSENT`. The same robot commanded remotely through
`robot_mesh` was gated, so a local agent moved the servos unasked while a remote
one had to wait for a human (F-011, CWE-862).

Both actions now run through the shared command gate before anything is
dispatched: `STRANDS_ROBOT_COMMAND_ALLOW` (`execute`, `start`, or `*`)
pre-approves, `BYPASS_TOOL_CONSENT=true` lifts the gate with a WARNING, otherwise
the operator is asked through the invoking agent's interrupt and the reply is
recorded on the audit log. An `AgentTool` receives no tool context, so the robot
builds one from the agent in `invocation_state` the way the SDK does for a
decorated tool; an unanswered question yields the same `ToolInterruptEvent` the
SDK resumes from, and with no agent reachable the call is refused. A grant the
dashboard's `MotionInterruptHook` deposited for the same call is spent instead of
asking twice, and an absent dashboard extra reads as "no grant", never a crash.
`status` and `stop` are never gated; the simulation tool is untouched.
