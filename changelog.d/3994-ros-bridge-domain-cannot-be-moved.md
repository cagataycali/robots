### Fixed: a ROS 2 bridge refuses a domain the process is not on

`RosTelemetryBridge` pinned its ROS 2 domain by writing `ROS_DOMAIN_ID` and then
initialized `rclpy` only when no context was running yet. A context reads that
variable once, at `init`, and keeps it for its lifetime, so in a process that
already had one -- a `rclpy` node embedding the simulation, or the hardware
bridge of the arm a sim mirrors -- the write was inert and the bridge published
on the other context's domain. Measured on ROS 2 Jazzy:
`SimRosBridge(domain_id=7)` beside a context on domain 0 reported success with
`ROS_DOMAIN_ID=7`, and a subscriber on domain 7 saw 0 messages while domain 0
carried 969 -- from two publishers on one `/so101/joint_states`, interleaving a
real arm's pose with the sim twin the operator had asked to isolate.

Such a request is now refused at construction, naming the domain in force and
both ways on (shut that context down, or pass that domain), and the domain write
is rolled back so the refusal leaves the environment as it found it -- the
contract the range and QoS refusals already kept. A bridge asking for the domain
already in force is unaffected.
