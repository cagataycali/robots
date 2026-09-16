### Fixed: the MoveIt2 sidecar names the install it needs instead of a traceback

`python -m strands_robots.policies.moveit2.server.zmq_node` launched in a shell
with no ROS 2 sourced used to die on a bare `import rclpy` - a traceback ending
in `No module named 'rclpy'` with the remedy nowhere in it. Every module the
sidecar imports lazily is now refused with the step that supplies it: pyzmq and
msgpack from the `[moveit2]` extra, and `rclpy`, `moveit_py`,
`moveit_configs_utils` and `geometry_msgs` from a system ROS 2 + MoveIt 2
install. The startup ones are refused before a socket is bound, as one error
line and exit status 2.

A MoveIt 2 that is present but whose binding moved is not an install to perform,
so it keeps its traceback and exit status 1: the gate raises its own
`MissingRosModuleError`, which `ImportError.name` cannot substitute for.
