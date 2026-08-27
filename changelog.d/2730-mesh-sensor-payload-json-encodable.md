### Fixed: a sensor payload's numpy readings are coerced, so the topic reaches the wire instead of being dropped

`SensorLoopsMixin`'s readers merged the robot's provider mapping into the outgoing
payload verbatim -- `summary.update(data)`, `imu.update(imu_data)`, and six more --
and every extended sensor topic is published as JSON: `_put_zenoh_directly`
encodes the payload before it reaches the wire. A payload the encoder refuses is
not a transient failure that the next tick retries; it fails identically forever,
which is why #2638 reports it at ERROR rather than absorbing it. The topic never
publishes at all.

The values a sensor stack reports are numpy. A lidar summary's bounding box is
whatever `ndarray.min(axis=0)` returned, an IMU's orientation a `float32`, a
device state code an `int64` -- and `json.dumps` refuses all three. So a robot
exposing `_lidar_summary`, `_imu`, `_odom`, `_lidar_state`, `_map_info`, `_hands`,
`_temps`, `_battery` or a `_pose`/`_slam_pose`/`_odom_pose` mapping built from its
own readings had those topics silently absent from the mesh, with one ERROR line
per topic to say so.

The defect turned on the numpy *width*, which is why it survived. `np.float64`
subclasses Python's `float` and `np.str_` subclasses `str`, so a payload built
from those always encoded; `np.float32`, `np.float16`, `np.int32`, `np.int64`,
`np.bool_` and `ndarray` subclass nothing and were refused. A robot on a float64
pipeline published fine, and the same code reading a `float32` point cloud
dropped every tick.

Two paths in the same class already coerced for exactly this reason -- `_read_imu`'s
inner-observation branch calls `tolist()`, `_read_pose`'s SE(3) branch calls
`float()` -- and so does `Mesh.publish_step`. The eight readers that merge a
provider mapping did not, so the class held two conventions for one wire format.
Each reader now hands its payload to one `_coerce_record` before returning it:
anything exposing `tolist()` becomes the equivalent Python list or number, and
containers are rebuilt with their contents coerced, mapping keys included, because
a `float32` is no more encodable as a key than as a value.

The boundary is that coercion repairs readings rather than laundering payloads. A
value that is not a reading -- an opaque object, a `bytes` blob -- is returned
untouched, so it still reaches the encoder and is still reported by name; and the
recursion is bounded, because a sensor record is shallow and a structure deeper
than the bound is a cycle or an object graph rather than a reading, which #2638's
report is the right answer for. Coercion also never edits robot state: `_read_health`
stores the provider's `_temps` mapping by reference, so the entry is replaced in the
payload rather than rewritten underneath the robot.

`tests/mesh/test_sensor_payloads_are_json_encodable.py` drives all eight readers
and all three pose branches with the numpy a sensor stack reports, asserting on the
decoded record a subscriber sees. Each row carries a premise that its raw provider
value is what the encoder refuses, so the table cannot pass on a payload that never
needed coercing. Thirteen plausible regressions were applied and the pin re-run:
twelve fire a different set, none goes undetected, and the control is clean. The
rule that every reader routes its payload through the coercion is derived from the
module rather than listed, so a reader added later is held to it on arrival.
