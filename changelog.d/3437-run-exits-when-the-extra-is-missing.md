### Fixed:
- `Robot(...).run()` without the `[device-connect]` extra now prints the install command, releases the robot and exits 1, instead of printing "Ctrl+C to stop" and sleeping forever on a process that serves no transport. A bring-up that fails for a possibly transient reason (broker unreachable) still parks as before.
