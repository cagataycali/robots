### Added: the Unitree B2 is driven natively over the Go2's `unitree_go` wire

`Robot("b2", mode="real", port="<robot IP>", network_interface="eth0")` now
builds `Go2Driver` instead of refusing the robot: lerobot registers no B2 type,
so it was simulation-only. The SDK's `example/b2/low_level/b2_stand_example.py`
uses the Go2's `unitree_go` `LowCmd_`, header, `LegID` slot order, mode `0x01`
and motion-switcher release, so the B2 is one more `WireProfile`
(`strands_robots.drivers.go2.WIRE_PROFILES["b2"]`) with the example's
`kp=1000, kd=10`. Joint targets are keyed by the description's names
(`FL_hip_joint`, ...), never by index.
