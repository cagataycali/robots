### Fixed: `Teleoperator()` names this host's serial devices when a leader has no port

`Teleoperator("so101_leader")` with no `port=` used to surface only lerobot's
`missing 1 required positional argument: 'port'`. It now says
`Pass Teleoperator('so101_leader', port=...)` and names this host's serial
devices with their usb ids, the same sentence `Robot(..., mode="real")` already
gives a follower with no port. Teleoperators with no `port` field (gamepad,
keyboard, phone) are unchanged. The scan-and-explain sentence now lives once in
`strands_robots._serial_discovery.describe_missing_port`, shared by all three refusals.
