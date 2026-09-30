### Fixed: the dashboard's tool factory and its motion gate read one precedence, metal first

`classify_peer` read `robot_type` first and `peer_is_physical` read `hw` first,
so a presence record naming both got a sim tool for a peer the gate called
metal, and since only real-arm tools entered the interrupt table that tool's
`execute` and `start` never reached the gate. Both now read
`agent_motion.hardware_evidence` first, and `motion_actions_for` takes the
peers so every proxy whose peer the gate calls metal (a wire sim claim this
dashboard did not launch included) is interrupted; a dashboard-launched sim
stays ungated and the sim tool surface is unchanged.
