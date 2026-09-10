### Fixed: `Robot(..., ros2_transport="rtps")` in sim mode is refused by name instead of asking for rclpy

The pure-RTPS transport is a hardware-bridge option; the simulation bridge is
rclpy-only. The sim backend absorbed `ros2_transport` through `**kwargs`, so a
caller who chose the rclpy-free transport was told "'rclpy' is required" - the
one dependency that choice was documented to avoid. `Robot(mode="sim")` now
raises `ValueError` naming the option as hardware-only and the two ways out
(`mode="real"`, or drop the argument).
