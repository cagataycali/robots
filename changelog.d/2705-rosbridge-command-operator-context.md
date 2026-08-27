### Fixed: a `RosbridgeRobot` command can carry an operator's decision

`use_rosbridge` gates a publish aimed at a safety-critical command surface, and `cmd_vel` is one.
`RosbridgeRobot.drive` and `.stop` took no `tool_context` parameter and their two agent tools were
not declared `@tool(context=True)`, so there was nothing for the gate to ask an operator with:
every command this bridge sent to a blocklisted topic hit the gate's fail-closed branch. Measured
against the real transport on the same `/cmd_vel`, with an operator standing by to approve:

    RtpsRobot.stop(tool_context=ctx)  ->  success, 1 message published
    RosbridgeRobot.stop()            ->  error, "No tool_context available for operator
                                         approval", 0 published

It was the only drive-owning class in the package with that gap - `RosBridgedRobot` and
`RtpsRobot` inherit the threading from the shared mobile base, and `AckermannRosRobot`, which
also owns its own `drive`/`stop`, threads it itself. The halt was the worst case: the one control
the `tools` contract guarantees for anything that can start motion ("a caller that can start
motion must be able to end it") was the one an operator who was present and willing could not
authorise. The only remedies the bridge could offer were the blanket `BYPASS_TOOL_CONSENT` or a
standing pre-approval, both of which give up the per-command decision the gate exists for.
`drive`, `stop` and the trailing zero of a timed drive now forward the context, the trailing zero
carrying the same one as the command it undoes - a zero that could not reach the gate would be
refused on its own and leave the robot latched at the speed of a hold the operator did approve.

The documentation said the opposite. `docs/rosbridge-integration.md` described the halt as
`**stop() never gated** (this bridge only)`, repeated it in two comments of a runnable example
(`# always publishes zero Twist, never gated`, `# single-shot, ungated`) and in a method table
row, and `RosbridgeRobot.stop`'s own docstring read "Never gated on anything." A reader who
believes that leaves `cmd_vel` out of `STRANDS_ROS2_COMMAND_ALLOW` and discovers in the field
that the halt is refused - the unreachable-halt hazard, arriving through documentation. All four
surfaces now state what is actually true of this bridge and every other: the halt needs no prior
state and no enable handshake, and it is not exempt from a gate that is keyed on the command
surface rather than on the payload. The shared mobile base already wrote that argument down; zero
means "stationary" on a `Twist` but commands motion to the zero pose on a joint-command topic, so
a payload-shaped carve-out could not be written correctly.

`tests/test_use_ros_command_blocklist.py` exists to catch exactly this - it scans operator-facing
prose for a clause that names the halt and denies it is gated, then grades the claim against the
running gate. Its document list was three hardcoded files keyed on naming the pre-approval
variable, and a per-transport integration page shows an operator how to drive `cmd_vel` without
ever mentioning that variable, so the page carrying all three false claims sat outside everything
the scan read. With the scan pointed at its old list and the prose left as it was, the guard
passes: the blindness is the silence. The scanned set is now derived - a page qualifies when it
names a blocklisted surface by the same final-segment rule the gate matches on - which takes it
from 3 pages to 9 and pulls in every transport integration page. The narrow list is kept for the
separate question it answers correctly, which pages document the variable itself, since its own
non-vacuity check requires each entry to name it. This is the second hardcoded universe in this
area to be derived after #2700 did the same for the drive-contract scope survey; the shape is a
guard whose discovery signature is narrower than the thing it guards.

A new `tests/mesh/test_rosbridge_robot_command_gate.py` is the fourth of four per-bridge gate
suites, doubling the roslibpy client beneath the real `use_rosbridge` so the gate and the
transport wiring both run unmodified. `tests/mesh/test_rosbridge_robot.py` could not have seen
the defect: it patches the `use_rosbridge` symbol at the mesh boundary, which is the boundary the
gate lives behind. Its structural half is derived over every drive-owning class, so a fifth
mobile base fails it on arrival rather than shipping unapprovable.
