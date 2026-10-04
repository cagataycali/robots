### Added: a Hello Robot Stretch native driver over `stretch_body`

`Robot("stretch3", mode="real")` (and `Robot("stretch", mode="real")`) now
builds `StretchDriver` (`strands_robots.drivers.stretch`) on the robot's own
computer instead of refusing the robot: lerobot registers no Stretch type, so
it was simulation-only. `send_action` queues lift, arm, wrist and head goals
(the Stretch 3 MuJoCo actuator names; metres and radians) and pushes them once;
`set_twist(vx, wz)` drives the base. Where the vendor's calls quietly clip a
goal to the soft limits, ignore an unhomed joint or clamp a wheel speed, the
driver refuses and names why; a twist is also refused when the wheels'
velocity watchdog is off. `state`, `run_policy`, `start_task` and `stop` are
wired. The gripper is not driven yet. The SDK ships on the robot
(`pip install hello-robot-stretch-body` elsewhere); it is not an extra.
