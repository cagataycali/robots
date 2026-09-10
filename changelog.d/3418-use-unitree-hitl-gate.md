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

The same finding reached the SDK's raw transport, which a name-keyed gate cannot
see. Every client inherits `_Call(apiId, parameter)` from
`unitree_sdk2py.rpc.client.Client` and each typed method is a thin wrapper over
it, so `_Call(7105, ...)` is the wire form of `loco.SetVelocity`. An underscore
name matched no mutative prefix and no danger pair, so it fell past the gate and
`_execute` dispatched it - the same walk, with no prompt and no audit row.
Private names are now refused before dispatch rather than gated, because a raw
call carries its command in an opaque `apiId` that no operator prompt and no
name-keyed table can judge; an approval, `BYPASS_TOOL_CONSENT` and a `*`
allowlist all leave the refusal standing. The refusal still carries `mutative`
and `high_danger`, so it cannot be read as a harmless call, the classification
fails closed so the fallback is a gate rather than a dispatch, and
`describe_operation` declines a private name instead of answering off the base
class that it is neither mutative nor high-danger.
