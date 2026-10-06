### Fixed: the dashboard agent asks before resetting or stepping a peer whose sim claim it cannot check

A mesh peer that announced `robot_type: "sim"` without this dashboard having
launched it was already treated as real hardware by the motion gate, but the
agent's proxy tool for it only asked before `execute` and `start`, so an agent
`reset` or `step` reached the robot unasked. Every proxy on a peer the gate calls
physical now asks before each verb in `agent_motion.PHYSICAL_MOTION_ACTIONS`
(`execute`, `start`, `teleop_receive`, `reset`, `step`), the set `GATED_ACTIONS`
is built from. The robot card reads the hardware marker first and shows its
confirm sheet for a wire sim claim the server could not corroborate.
