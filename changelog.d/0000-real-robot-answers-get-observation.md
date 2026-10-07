### Fixed: `Robot(name, mode="real")` answers `get_observation()`, as `mode="sim"` does

A loop written against the simulation, `obs = robot.get_observation()`, raised
`AttributeError` once `mode="real"` was set, on both real paths. The lerobot
wrapper (`strands_robots.hardware_robot.Robot`) now forwards the call to its
lerobot device, connecting on first use with `calibrate=False` as `send_action`
does, and the native Feetech driver (SO-100, SO-101, Koch through
`DynamixelDriver`) answers it with every joint as `"<joint>.pos"` plus one frame
per open camera. Both return lerobot's keys, not the simulation model's joint
names, and both raise when the port will not open instead of answering an empty
dict. The SO-arm and LeKiwi robot pages now say their observation table lists
`mode="sim"` keys. The mesh camera publisher keeps reading a native driver's
cameras directly, so publishing video never opens the motor bus.
