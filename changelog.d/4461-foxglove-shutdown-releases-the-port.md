### Fixed: a stopped Foxglove bridge has released its port when `shutdown()` returns

`FoxgloveBridge.shutdown()` (and `Robot.cleanup()` on a robot started with
`foxglove=`) used to return while the SDK's listener thread still held the
socket, so for a few milliseconds a stopped bridge accepted a connection and
dropped it, and a new bridge on the same fixed port could find it busy. It now
waits, up to two seconds, until the port can be bound again.
