### Fixed: a Spot without `bosdyn-client` is told to install the extra, not to set credentials

`SpotDriver.connect_eagerly` checked `BOSDYN_CLIENT_USERNAME` and
`BOSDYN_CLIENT_PASSWORD` before it imported the SDK, so a fresh install was
sent to set two variables that could not help. It now names
`pip install 'strands-robots[spot]'` first, as the xArm, Crazyflie, RB-Y1 and
UR drivers do; the credentials check still runs once the SDK imports.
