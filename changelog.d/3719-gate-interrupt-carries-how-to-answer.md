### Fixed: a paused real-robot call tells the script how to resume it

`agent("rotate the wrist 5 degrees")` on `Robot(mode="real")` returns a PAUSED
result - the operator gate raised an interrupt and the SDK handed it back to be
answered - but a script that printed it saw the interrupt list's repr: the
action, the target, the warning, and nothing that said the run was paused or how
to continue. Every interrupt raised through the shared command gate (`robot`,
`serial_tool`, `pose_tool`, `use_unitree`, the ROS transports) now carries
`how_to_answer` in its `reason`: that the call is paused with the question in
`result.interrupts`, the exact
`agent([{"interruptResponse": {"interruptId": ..., "response": "y"}}])` resume
form, that anything but `y` denies and nothing moves, and the exact
`..._COMMAND_ALLOW=<value>` that pre-approves this command for a script with no
operator. That value is read from the tool's own allowlist matcher, because the
spelling it accepts is per-tool - an action name for `robot`/`serial_tool`/
`pose_tool`, a `service.operation` pair for `use_unitree`, a surface name for
the ROS transports. The existing `action`/`target`/`warning` fields and the
headless refusal are unchanged. `docs/security.md` gains "Answering a command
gate from a script".
