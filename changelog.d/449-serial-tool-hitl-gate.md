### Fixed: `serial_tool` asks an operator before a write reaches the servo bus

The four write actions - `send`, `send_read`, `feetech_position`,
`feetech_velocity` - reached `ser.write` with no gate of their own. The
dashboard's `MotionInterruptHook` lists them, but a hook is something an `Agent`
has to be built with, and every canonical `Agent(tools=robot.tools)` build has
none, so the default wiring moved servos on whatever steered the agent (F-009,
CWE-862). Its comment also pointed at a `dashboard/direct_serial.py` that does
not exist.

`serial_tool` now takes the operator context (`@tool(context=True)`) and each
write action runs through the shared command gate after the option check and
before the port is opened: `STRANDS_SERIAL_COMMAND_ALLOW` (action names, or `*`)
pre-approves, `BYPASS_TOOL_CONSENT=true` lifts the gate with a WARNING, otherwise
the operator is prompted and the reply is recorded on the audit log; with no
context reachable the call is refused and nothing is written. A grant the
dashboard hook already deposited for this exact call is spent instead of asking
twice, and an absent dashboard extra reads as "no grant", never a crash. Reads,
`monitor` and `feetech_ping` are never gated. The gate's transport-agnostic half
lives in `_command_gate.gate_motion`, shared with the ROS transports, which
now reach the operator through it too.
