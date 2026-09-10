### Fixed: `use_unitree` asks an operator before a mutative RPC reaches the robot

The raw Unitree SDK2 escape hatch dispatched `loco.SetVelocity`, `loco.Move`,
`loco.ZeroTorque`, `loco.SetFsmId` and `motion_switcher.ReleaseMode` with a
`logger.warning` as its only rail, while the sibling ROS transports refused the
same class of command without a human's `y`. An agent steered by untrusted
content - a message, a retrieved document, another tool's output - could walk a
standing humanoid or drop its holding torque with nobody in the loop (F-001,
CWE-862).

`use_unitree` now takes the operator context (`@tool(context=True)`) and every
mutative or high-danger operation runs through the shared command gate before
`_execute` touches the bus: `STRANDS_UNITREE_COMMAND_ALLOW` (exact
`service.operation` entries, or `*`) pre-approves, `BYPASS_TOOL_CONSENT=true`
lifts the gate with a WARNING, otherwise the operator is prompted and the reply is
recorded on the audit log; with no context reachable the call is refused and the
envelope says `dispatched: false`. Reads, the `meta` operations and
`loco.StopMove` are never gated. The gate's transport-agnostic half now lives in
`_command_gate.gate_motion`, which `gate_command` fronts with its ROS blocklist,
so a Unitree RPC and a ROS publish share one interrupt site and one audit row.
