### Added: a live Foxglove view and MCAP export for any robot

`Robot(name, foxglove=True)` serves a simulated or real robot to Foxglove over
the `foxglove.sdk.v1` WebSocket protocol: `/tf` and the robot's own MuJoCo
meshes for the 3D panel (no URDF), per-robot joint states, JPEG cameras, and
log and event channels, on the same telemetry path the ROS 2 bridges use.
`foxglove_mcap=path` records the same channels to a new MCAP file (the static
scene once), `strands_robots.foxglove.export_episode` turns one LeRobot v3
episode into a seekable MCAP, and `mcap_info` reads any MCAP back. The server
advertises no capability unless `foxglove_services=True`, and the one service
it then offers is refused until `STRANDS_FOXGLOVE_COMMAND_ALLOW` names it.
`STRANDS_ROBOTS_FOXGLOVE=1` switches the view on for every `Robot()`. New
`[foxglove]` extra, folded into `[all]`; one Learn page under Data.
