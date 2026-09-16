### Fixed: the quickstart says what step 5 needs

The closing line claimed everything past the hardware and GPU steps runs in
sim; step 5's ROS 2 bridge needs a sourced ROS 2 distro (`rclpy` is not on
PyPI). The line now names the requirement and the remedy the error gives.
