### Fixed: `Robot.run()` exits 1 when Device Connect refuses an unauthenticated transport, instead of parking

The refusal is raised as `DeviceConnectRefused` (a `RuntimeError` subclass, so
broad handlers keep working) and the foreground loop treats it like a missing
extra: print the cause, release the instance, exit 1. A broker outage still parks.
