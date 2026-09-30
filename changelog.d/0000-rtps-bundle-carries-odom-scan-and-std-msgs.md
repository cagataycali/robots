### Fixed: `use_rtps` and `RtpsRobot` speak `/odom`, `/scan` and `std_msgs`

The pure-RTPS bundle had 9 types, so `use_rtps` refused `std_msgs/msg/String`,
`nav_msgs/msg/Odometry` and `sensor_msgs/msg/LaserScan` as "not in the RTPS IDL
bundle". That included the `echo /odom` the ROS 2 page opens with. It now also
carries the `std_msgs` `String`, `Bool`, `Int32`, `Float32` and `Float64`;
`PoseStamped`, `TwistStamped`, `PoseWithCovariance` and `TwistWithCovariance`;
`nav_msgs/msg/Odometry`; and `sensor_msgs/msg/LaserScan` and `Imu`. Each is laid
out as ROS 2 Jazzy's `.msg` and was checked on the wire against a real ROS 2
node, in both directions. `RtpsRobot` takes `odom_topic` / `scan_topic` (typed
`Odometry` / `LaserScan` by default), so `get_pose` and the `get_pose_<name>`
tool work. A `publish` whose values do not fit the type (`{"linear": 3}` for a
`Twist`) is refused by name before a writer joins the graph. It used to raise
cyclonedds' bare `Exception` out of the tool.
