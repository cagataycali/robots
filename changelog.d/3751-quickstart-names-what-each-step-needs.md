### Fixed: the quickstart says what each step of the whole loop needs

The closing line covered two of the five steps and swept the rest into
"everything else runs in sim". Step 5's ROS 2 bridge needs a sourced ROS 2
distro (`rclpy` is not on PyPI) and step 4's mesh call needs the `[mesh]` extra
plus a mesh posture, neither of which the install command on the page provides.
The line now names both requirements and the remedies the library gives.
