### Fixed: `robot_mesh` read-only actions say when no discovery ran

`action="peers"` and `action="status"` answer before the outbound-mesh gate, so a
process with no local `Robot()`/`Simulation()` reported `0 local, 0 remote` with
`status="success"` even when the robot-less gateway peer it hears the fleet
through never came up -- byte-identical to a real discovery that found nothing.
Under `STRANDS_MESH=false` the report also offered "create a `Robot()`", advice
the same kill switch refuses, which is the reasoning every other action already
applies when it names the variable. Both actions now carry `no discovery ran`
beside the count, naming `STRANDS_MESH` when the switch is the reason and
pointing at the DEBUG bring-up failure otherwise, and audit the call as
`discovery=none`. A gateway that came up, or a process with its own mesh, reports
the same measured counts as before.
