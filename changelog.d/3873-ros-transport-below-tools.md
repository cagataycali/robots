### Changed: the in-process rclpy transport sits below both robots that publish through it

`RosBridgedRobot` and `AckermannRosRobot` opened a publisher by importing the
`use_ros` `@tool`, so two library classes reached up into the agent-tool layer
for their transport. The rclpy mechanics - the process-wide node and executor,
the dynamic type resolution, the graph introspection, the pub/sub/service calls
and the action-goal lifecycle with its timeout cancel - now live in
`strands_robots.ros`, in the same layer as the two robots; `use_ros` keeps what
an agent envelope owns (the numeric-option domains, the operator gate, the
docstring a model reads). `ros_action` takes the operator gate as a required
keyword-only argument, so no caller can command a blocklisted surface by
forgetting one, and all three callers key it with the same label, so a `cmd_vel`
publish files one interrupt id and one audit source whichever surface asked. No
behaviour a caller sees changes.
