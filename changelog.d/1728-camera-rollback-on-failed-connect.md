### Fixed: a failed connect no longer leaves a camera it already opened streaming

A robot's `connect()` opens its devices in sequence - the motors bus, then each
camera in turn - and nothing in that loop closes the cameras opened before one
that fails. The connect-failure rollback closed only the serial port, so a
camera set was left half-open, and lerobot gates both recovery paths on
`is_connected`: the retry raised `DeviceAlreadyConnectedError` on a camera that
was healthy, masking the camera that actually failed, and `disconnect()` refused
to run at all while one camera was still shut. A second attempt therefore
reported a generic "Failed to connect" naming no device, and every later attempt
kept reporting it. The rollback now closes every device that is open, each
independently and best-effort, so the retry keeps surfacing the real fault and
no device node stays held.
