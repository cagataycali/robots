### Fixed:

- The downloaded Amazon Root CA1 copy under `~/.strands_robots/iot/` is now created owner-only (mode 0o600) in one step, like the cert and key next to it; it used to be written and then loosened to 0o644 (CodeQL py/overly-permissive-file).
- The seven py/path-injection findings on `DatasetRecorder.create`'s target check are closed by feeding it a value CodeQL can see was contained to the LeRobot dataset home.
