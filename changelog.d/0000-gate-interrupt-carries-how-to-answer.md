### Fixed: a paused real-robot call tells the script how to resume it

`agent("rotate the wrist 5 degrees")` on `Robot(mode="real")` returns a
PAUSED result - the operator gate raised an interrupt and the SDK handed it
back to be answered - but a script that printed it saw the interrupt list's
repr: the action, the target, the warning, and nothing that said the run was
paused or how to continue (the README's four-line script ends there). Every
interrupt raised through the shared command gate (`robot`, `serial_tool`,
`pose_tool`, `use_unitree`, the ROS transports) now carries `how_to_answer` in
its `reason`: that the call is paused with the question in `result.interrupts`,
the exact `agent([{"interruptResponse": {"interruptId": …, "response": "y"}}])`
resume form, that anything but `y` denies and nothing moves, and the tool's
`*_COMMAND_ALLOW` variable for a script with no operator. The existing
`action`/`target`/`warning` fields and the headless refusal are unchanged.
`docs/hardware/robot-control.md` lists all eleven real-mode actions with their
gating (the table had only four) and shows the resume loop; `docs/security.md`
gains "Answering a gate from a script".
