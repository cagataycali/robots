### Fixed:
- `strands-robots doctor` no longer fails (exit 1) a sim-only Linux user for not being in `dialout` when no serial device is plugged in; that is now a WARN with the same `usermod` remedy. It still FAILs when a `/dev/ttyACM*`/`ttyUSB*` device is present and unreadable, and PASSes when a udev rule already grants access.
