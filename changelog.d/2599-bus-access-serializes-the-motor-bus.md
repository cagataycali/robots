### Fixed

- **mesh**: one reader at a time on a robot's motor bus. A serial bus is a single
  conversation, and a mesh peer had four threads reaching for the same device --
  the state probe, the hardware camera publisher, the sensors probe and the IoT
  camera offload -- plus teleop writes. The feetech/dynamixel SDKs refuse a
  collision outright (`[TxRxResult] Port is in use!`), so on real hardware the
  reads collided continuously and the peer published no joints at all. Every
  reader and writer now goes through `strands_robots.bus_access`, which keeps one
  `RLock` on the DEVICE so wrappers in different modules share it. The state
  probe also reads the joints DIRECTLY, because a camera raising inside
  lerobot's `get_observation()` used to discard the joint positions it had
  already read. A bus left flagged in-use by an exchange that never finished now
  recovers on the next read, once, under the lock, and the peer publishes
  `bus_recoveries` so a degrading cable is visible instead of silently healed.
