### Docs: the ROS 2 row says a running sim needs a sourced distro

The README listed "expose a running sim" next to `use_rtps` ("without rclpy"),
so a reader on `pip install 'strands-robots[sim-mujoco,ros2]'` with no ROS 2
distro expected `Robot("so100", ros2_bridge=True)` to work and got an
`ImportError`. The sim bridge (`SimRosBridge`) is rclpy-only and the sim takes
no `ros2_transport`; only a real arm has the pure-DDS path. The README row and
the ROS 2 page now say so.
