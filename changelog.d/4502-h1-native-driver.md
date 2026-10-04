### Added: the Unitree H1 is driven natively over the Go2's `unitree_go` wire

`Robot("h1", mode="real", port="<robot IP>", network_interface="eth0")` now
builds `Go2Driver` instead of refusing the robot: lerobot registers no H1 type,
so it was simulation-only. The H1 speaks the same `unitree_go` `LowCmd_`, header,
CRC and motion-switcher release as the Go2, so it shares that driver through a
per-robot `WireProfile` (`strands_robots.drivers.go2.WIRE_PROFILES`): the H1's
19 joints are keyed by the MuJoCo model's names onto the SDK's `H1JointIndex`
slots, the ankles and arms take mode `0x01` with `kp=60, kd=1.5`, the leg and
torso motors mode `0x0A` with `kp=200, kd=5`. A joint name of one robot sent to
the other is refused, never written. `Go2Driver(model="h1")` picks the H1
profile for a renamed mesh peer. The H1-2 (`unitree_hg`) is not covered.
