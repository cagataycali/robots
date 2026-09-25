### Changed: the G1's six snapshot readers are one `g1_sensor` verb

`g1_battery`, `g1_imu`, `g1_lidar_state`, `g1_lidar_summary`, `g1_mainboard` and
`g1_pressure` read six caches the driver's own DDS subscribers write, differing
only in which cache attribute they read and which fields it carries.
`g1_sensor(driver, sensor)` reads the same six through one table, with every
per-sensor envelope unchanged, and the fields are derived from the driver's
writers rather than restated. `tools/` publishes 63 verbs over 31 files, down
from 68 over 36.
