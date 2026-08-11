### Fixed: `RosBridgedRobot` now exposes a `stop_<node>` agent tool

A ROS 2 velocity command latches: `drive()` with no `duration` publishes a single
`cmd_vel` message and the base keeps it until something else is published -- the
contract `RosbridgeRobot`'s own drive tool description states as "without
duration the last command latches until stop". All three mobile-base bridges
carry a public `stop()` that publishes a zero `Twist`, and the rosbridge and
RTPS bridges expose it as a named agent tool. The ROS 2 bridge did not, so an
agent handed `RosBridgedRobot.tools` could start motion and had no tool to end
it; its only halt was a `drive` at zero velocity, which is the idiom `stop()`
exists to name.

`tools` now builds `stop_<node>` unconditionally, alongside `drive`, so the
three transports agree on the capability an agent is given. A structural test
requires any future bridge owning a `drive`/`stop` pair to expose the halt too.
