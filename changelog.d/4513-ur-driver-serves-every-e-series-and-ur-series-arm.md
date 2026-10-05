### Added: `URDriver` serves every e-Series and UR-Series arm, at each model's own joint speeds

`Robot("<model>", mode="real", port=...)` now builds the native RTDE driver for
`ur3e`, `ur7e`, `ur12e`, `ur16e`, `ur8long`, `ur15`, `ur18`, `ur20` and `ur30`
as well as `ur5e` and `ur10e`; before, those nine answered with lerobot's list
of robot types. The per-joint speed ceilings the step gate uses are copied from
each model's `joint_limits.yaml` in Universal_Robots_ROS2_Description, which also
corrects the UR10e's elbow from 120 to 180 deg/s (only its base and shoulder are
held to 120). A name the table does not carry is held to the slowest ceiling any
served model has on each joint (`FALLBACK_SPEED_RAD_S`, replacing
`FALLBACK_MODEL`). CB3 arms (`ur3`, `ur5`, `ur10`) stay unserved.
