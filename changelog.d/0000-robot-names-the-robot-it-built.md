### Added: every `Robot()` answers `robot_name`, and a simulation's repr names its robot

`Robot("so100").tool_name` is the agent-tool name, `"so100_sim"` in simulation,
which every sim method refuses as a robot name, and the engine had no other
name to read back: `repr` was the default `<... object at 0x...>`. The return
of `Robot()` now carries `robot_name`, the string it was built with
(`"so100"`), on the simulation engine and on every real-hardware driver alike,
so a caller threads the same name through `robot_joint_names(...)` in either
mode. A simulation answers it from its world (`None` when it holds several
robots) and renders as `<MuJoCoSimEngine robot='so100' tool='so100_sim'>`.
`tool_name` is unchanged.
