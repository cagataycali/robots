### Fixed: a remote reset or step needs operator approval, and the approval is bound to the peer that sent it

A real robot on the mesh ran a wire `reset` (every joint to its home pose) or
`step` with no approval, and approved `execute` / `start` / `teleop_receive`
for anyone who could publish. Those five verbs now share one definition with
the dashboard gate (`_command_gate.PHYSICAL_MOTION_ACTIONS`), and a hardware
peer refuses a motion command unless the sample's publisher session is the one
the named sender announced its presence from. An approved teleop stream follows
that one session and ends after `STRANDS_MESH_INPUT_STREAM_TTL_S` (900 s by
default). Device Connect `execute` from an allowlisted caller now also needs
the operator's approval (`STRANDS_ROBOT_COMMAND_ALLOW=execute` or a dashboard
grant). Stopping is unchanged.
