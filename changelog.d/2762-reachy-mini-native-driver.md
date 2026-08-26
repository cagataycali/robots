### Added: native daemon driver for the Pollen Reachy Mini

`Robot("reachy_mini", mode="real")` raised `ValueError: Unsupported robot type:
'reachy_mini'` and listed lerobot's known types. That was not a gap in the
registry - the Mini has been in it for some time, with a MuJoCo asset and four
aliases - but lerobot has no robot class for it, and the default driver builds a
lerobot `RobotConfig`. There was no way to reach the hardware at all.

The Mini is the second shape the driver seam was built for, and it is a different
shape from the first. The G1 needed a bus lerobot does not speak; the Mini needs a
*robot model* lerobot does not have. It is an expressive desk robot - a 6-DOF
Stewart head on a rotating body, two antennas, a speaker, a recorded-emotion
library - with no arms and no gait, addressed through the Reachy daemon's REST API
on `:8000` plus a real-time link. Its state of interest is head orientation, not a
joint-space arm pose. `Robot("reachy_mini", mode="real")` now builds
`strands_robots.drivers.reachy.ReachyDriver`, because the registry declares
`hardware.driver = "strands"` and `driver="auto"` honours that declaration. The
entry carries no `lerobot_type`, which is the honest statement of the situation:
`driver="lerobot"` is still refused by name.

`connect_eagerly` probes `GET /api/daemon/status`, which is both the reachability
check and the call that reports which hardware variant answered - a Lite (no
onboard computer) is driven over a WebSocket to the daemon, a Wireless (onboard
CM4) over Zenoh. Both links, and the REST helper, come from
`strands_robots.device_connect.reachy_transport`, which the Device Connect driver
already ships; this driver reuses them rather than growing a second daemon client.
Off hardware the probe cannot reach a daemon and the returned reason names the
address that did not answer, leaving the driver disconnected but usable, so a mesh
peer for a Mini that is switched off is still constructible.

The sensor attributes `strands_robots.mesh.sensors.SensorLoopsMixin` publishes are
filled from what the link delivers: `_imu` is the head IMU verbatim, `_pose` is the
head orientation, and `_battery` is read from the status payload when it carries
one. There is no `_lidar_*`, because the Mini has no lidar.

`_pose` is taken from the IMU quaternion rather than derived from the six leg
positions the link also reports. The Mini's IMU is mounted *in the head*, so its
quaternion is the head's orientation as measured; converting leg positions into a
head pose would need Stewart-platform forward kinematics this repository does not
have, and inventing them would put a number on the mesh that nothing measured. An
IMU frame carrying no quaternion therefore leaves `_pose` alone instead of
publishing a pose with no orientation in it. The legs are still reported, as legs.

The motion envelope is a separate module, `strands_robots.tools.reachy`, because
two consumers need the same answer: this driver and the `reachy_*` agent tools that
will sit on the same daemon. The limits are not all of one kind. Pitch and roll are
the platform's own travel; the head-body yaw delta is a *coupling* limit between
two values, and neither value alone is out of range when the pair is - a head at
+60 and a body at -60 are each individually legal and together ask for a
120-degree twist the neck cannot make. A guard that checks one axis at a time
cannot see that, so the pair is checked as a pair.

The envelope refuses rather than clamps, and names the limit it refused against. A
clamp is a silent default, which this project's conventions forbid; on a robot it
is worse than a refusal, because the call reports success while the head goes
somewhere other than where it was sent. A caller therefore learns the envelope
from the refusal rather than from documentation.

`start_task` and `run_policy` refuse outright rather than standing in for work in
progress. With no arms and no gait there is no action space a policy is trained
against, so the refusal names the recorded-move path instead of implying a rollout
is coming. `stop`, by contrast, is real: `POST /api/move/stop` halts a recorded
move mid-play, and a daemon that refuses the stop is not reported as a halt.

Two statements in `docs/getting-started/robot-factory.md` were already false before
this change and are corrected here. The page said the lerobot driver "is what every
robot in the shipped registry uses", which stopped being true when the first native
driver landed, and it quoted a refusal ending "Robots with a native driver: none."
- that sentence is the live output of `list_native_drivers()`, so it was already
naming the wrong set.
