### Added: native CycloneDDS driver for the Unitree G1

The driver seam from the previous release chose an implementation but no native
driver was registered - every `mode="real"` call built a lerobot driver, so a
robot lerobot cannot model was reached through a bus it does not speak. The G1
is the shape that made the seam worth building: raw Unitree IDL over
CycloneDDS, a state machine that gates arm and locomotion writes, a Livox
Mid-360 producing a point cloud lerobot's dataset schema has no place for.
`Robot("g1", mode="real")` now builds `strands_robots.drivers.g1.G1Driver`,
because the registry declares `hardware.driver = "strands"` on `unitree_g1`
and `driver="auto"` honours that declaration.

The driver subscribes `rt/lowstate`, `rt/lf/bmsstate`, `rt/utlidar/lidar_state`
and `rt/utlidar/cloud_livox_mid360` on its own thread and fills the sensor
attributes `strands_robots.mesh.sensors.SensorLoopsMixin` publishes: `_imu`,
`_battery`, `_lidar_state` and a bounded `_lidar_summary`. The mesh reads
each with a `getattr` default, so a driver that has not yet received a topic
publishes nothing on it rather than a stale value, and the fleet dashboard's
chips populate as the callbacks deliver. The Livox cloud is summarised, not
shipped raw: a full frame is ~30k points at 10 Hz, and publishing that on
`lidar/summary` would drown Zenoh. The point-cloud tile lands separately (a
paced `lidar/cloud` topic and a Three.js viewer), and this driver's job is
the transport, not the render.

Writes are gated but not wired: `send_action`, `start_task` and `run_policy`
return a named "not wired yet" envelope in the same shape a rejected motion
call will one day return, and the two gates that will guard it - the FSM
must be in a scope-appropriate subset and the battery must be above 15% -
already consult the caches the DDS callbacks fill. The FSM gate carries two
sets rather than one, because the G1 documents them separately:
`HANDSHAKE_FSMS = {500, 501, 801}` covers arm-SDK writes (sitting accepts an
arm gesture) and `WALK_FSMS = {501, 801}` covers locomotion (sitting refuses
a walk). `send_action` is arm-scoped and names "arm writes" in its refusal;
`start_task` and `run_policy` are motion-scoped (their loops may issue
either kind, and the caller does not classify itself) and name the union;
the day the write lines land, the per-step call can pass `"loco"` and the
sitting refusal is already correct. Shipping the gates now means the day
the motion path lands the gates already have coverage, and a caller who
polls `send_action` today gets the honest reason for the refusal rather
than a stub that would look like a successful write.

`unitree_sdk2py` is lazy-imported: `from strands_robots.drivers.g1 import
G1Driver` never touches the SDK, so a headless CI machine can build the
driver, list it in the registry and run every unit test with a mocked bus.
The SDK only loads inside `connect_eagerly`, and a machine without it gets a
named reason back - the driver is left in a "usable but not connected"
state, which is what lets a mesh peer for an offline robot still be
constructed for the dashboard's "offline" card. Hardware bring-up on the
real G1 is validated at the office; Thor stays out of it.

The vendored `_dds_engine` and `_g1_common` under `strands_robots.tools.g1`
are the pieces the agent tools (`g1_arm`, `g1_locomotion`, `g1_speak`, ...)
will import too, and they share this module's `_DDS_INIT_LOCK` and
`ensure_dds` singleton so the driver and the tools never subscribe the Livox
cloud twice.
