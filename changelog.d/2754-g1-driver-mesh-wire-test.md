### Tests: a native G1 driver's DDS readings are pinned reaching the mesh wire

A native driver and the mesh sensor loops are joined by nothing but attribute
names. `SensorLoopsMixin` reads sixteen underscore-prefixed attributes off
whatever object the mesh was handed, and `G1Driver` fills four of them from its
DDS callbacks; nothing declares that relationship, so a rename on either side is
silent. The coverage that existed drove the readers through a host object whose
`publish` records the payload, which cannot see either the composition or the
encoding.

This composes a real `Mesh` around a real `G1Driver` fed mocked DDS and asserts
at the transport boundary that `strands/<peer>/lidar/summary`, `imu` and `health`
arrive and decode. The readings are fed as the numpy the SDK reports - a Livox
header is `int64`, an IMU orientation `float32`, and `json.dumps` refuses both -
so the cells grade the payload the robot really sends rather than a
pre-sanitised stand-in. A non-vacuity class shows the topics are the DDS samples
and not the loop merely running: an unfed driver publishes no imu or lidar
topic, while health still publishes because it aggregates host metrics too.

The loops are driven one tick each without threads or sleeping, because
`SensorLoopsMixin._paced` yields before it waits and so an already-set stop
event produces exactly one iteration.
