### Fixed: the MoveIt2 sidecar names the install it needs instead of a traceback

`python -m strands_robots.policies.moveit2.server.zmq_node` launched in a shell
with no ROS 2 sourced used to die on a bare `import rclpy` - a traceback ending
in `No module named 'rclpy'` with the remedy nowhere in it. Every absence
(pyzmq/msgpack from the `[moveit2]` extra, `rclpy`, `moveit_py` and
`moveit_configs_utils` from the system ROS 2 + MoveIt 2 install) is now refused
before a socket is bound, with one error line naming the step that supplies the
module, and exit status 2.
