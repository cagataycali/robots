### Added: the v1 types and `Description`

`strands_robots` now exports the data that flows between a world, a robot and a policy
(`Observation`, `Action`, `JointState`, `Pose`, `Calibration`, `Contact`, `Physics`,
`Requirements`, `RunResult`) and `Description.load(name)`, which resolves any of the 76
registry robots, a `robot_descriptions` name or an MJCF/URDF path to its joints, `action_keys`,
limits and keyframes. A registry robot with no simulation model refuses by name.
