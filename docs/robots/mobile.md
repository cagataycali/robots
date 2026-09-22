---
description: Quadrupeds, wheeled bases, mobile manipulators, and quadcopters.
---

# Mobile, mobile manip, and aerial

Quadrupeds, wheeled bases, mobile manipulators, and quadcopters.

```python
from strands_robots import Robot
sim = Robot("unitree_go2")      # Unitree Go2 quadruped
sim = Robot("spot")             # Boston Dynamics Spot
sim = Robot("stretch3")         # Hello Robot Stretch 3 (mobile manip)
sim = Robot("crazyflie")        # Bitcraze Crazyflie 2 quadcopter
```

## Catalog

Every robot in this family, generated from `robots.json` at build time. Renders are MuJoCo sim renders, never hardware photos.

{{robot_cards:mobile, mobile_manip, aerial}}

## Flying a real Crazyflie

`crazyflie` declares `hardware.driver = "strands"`, so `mode="real"` builds the native
CRTP driver over a [Crazyradio](https://www.bitcraze.io/products/crazyradio-2-0/) dongle.
lerobot has no robot type for a Crazyflie, so this is the only way to fly one from here.

```python
from strands_robots import Robot

cf = Robot("crazyflie", mode="real", port="radio://0/80/2M/E7E7E7E7E7")

# Opens the link and WAITS for the aircraft to answer, then ARMS the platform and
# starts telemetry. Returns None on success, or a reason - check it: nothing below
# can fly if the link never came up.
if (reason := cf.connect_eagerly()) is not None:
    raise SystemExit(reason)

# Every flight verb answers with an envelope, including for a link that went quiet
# after connecting - so read `status` rather than assuming the write landed. Each
# step is checked before the next is issued: after a refused takeoff there is no
# altitude for the twist to hold.
try:
    env = cf.takeoff(height=0.5, duration=2.0)
    if env["status"] == "success":
        env = cf.set_twist(vx=0.2, wz=1.0, z=0.5)  # 0.2 m/s fwd, 1.0 rad/s yaw, 0.5 m
    if env["status"] == "success":
        env = cf.land()                            # descends under control
    if env["status"] != "success":
        print(env["content"][0]["text"])
finally:
    cf.cleanup()  # lands if it is still flying, then releases the radio
```

Install the client library with the `crazyflie` extra:
`pip install "strands-robots[crazyflie]"`. It is **not** part of `[all]` - `cflib` is
GPLv3 and this project is Apache-2.0, so the copyleft dependency is only installed by a
caller who names it. Without it the driver still imports and registers; it reports a
reason naming this extra instead of connecting.

Five things behave differently from a ground robot, and each one is a way to break the
aircraft if you assume otherwise:

| | What to know |
|---|---|
| **Connecting waits** | `cflib`'s `open_link` is asynchronous and never raises - it reports failure on a *callback*, so an absent dongle or a switched-off aircraft returns normally. `connect_eagerly()` therefore blocks until the aircraft answers (up to `CONNECT_TIMEOUT_S`, 10 s) and returns the reason if it does not. Check that return: while there is no link `cflib` discards every packet in silence. |
| **Units** | `wz` is **rad/s**, as everywhere else in this package. `cflib` wants deg/s, and the driver is the only place that conversion happens. |
| **Setpoints are a stream** | The firmware supervisor cuts thrust when the setpoint stream goes quiet, so one `send_action` latches a setpoint and a background repeater keeps it alive at `setpoint_hz` (default 20 Hz). It returns when the setpoint is latched, not when the motion is done. |
| **`stop` lands** | `stop()` / `stop_task()` / `cleanup()` all perform a controlled descent. Cutting the motors - an airborne aircraft *falls* - is the separately named `emergency_stop()`, and the agent tool schema cannot reach it. |
| **A link can go quiet in flight** | `is_connected` reads True for a link that opened and then stopped answering - the handle is live and only the write finds out. So every flight verb returns an error envelope for a write the radio would not carry, rather than raising: check `status` on `send_action` / `set_twist` / `takeoff` / `land` / `emergency_stop`, not just on `connect_eagerly()`. Two of those refusals carry a consequence worth acting on - a refused priority handover means the climb or descent was *not* commanded, and a refused `emergency_stop` means the motors are **still turning** with the setpoint stream already stopped, so only a hardware cutoff will stop the aircraft. |

The flight envelope is the driver's, not the SDK's: `cflib` imposes no ceiling and the
firmware attempts whatever arrives. A setpoint outside it is **refused by name**, never
clamped, so a caller who asked for 5 m/s never silently flies 1 m/s. Read the bounds with
`strands_robots.drivers.crazyflie.twist_envelope()`.

Commands go through `send_action` / `set_twist` / `takeoff` / `land`; `start_task` and
`run_policy` refuse, because this package registers no aerial policy provider and a
quadcopter has no joints for a manipulation policy's action to land on. Telemetry
(`stateEstimate` position, `stabilizer` attitude, `pm.vbat`) is cached for the mesh; a bare
Crazyflie has no ranger deck, so no lidar topic is published.

## Real hardware: the Go2 native driver

The Go2 has no lerobot robot type, so `mode="real"` builds the native CycloneDDS
driver in `strands_robots.drivers.go2`. Its registry entry declares
`hardware.driver = "strands"`, so no `driver=` keyword is needed:

```python
from strands_robots import Robot

go2 = Robot("go2", mode="real", port="192.168.123.161", network_interface="eth0")
go2.connect_eagerly()          # subscribes rt/lowstate and rt/sportmodestate
go2.release_sport_mode()       # hands the legs over - see below
go2.send_action({"FL_calf_joint": -1.5})
```

The driver talks CycloneDDS through `unitree_sdk2py`, a vendor SDK that is not
an extra of this project; the install recipe per platform is in
[Installing the Unitree SDK](humanoids.md#installing-the-unitree-sdk), and a
missing SDK is refused with that recipe rather than only its module name.

Two Go2 specifics are worth knowing before writing a controller.

**Sport mode must be released first.** The Go2 ships with an onboard sport-mode
service driving the legs. Until it is released, a `rt/lowcmd` frame puts that
controller and your commands on the same twelve motors, so every write path
(`send_action`, `run_policy`, `start_task`) refuses until `release_sport_mode()`
confirms the robot reports no active mode. Releasing is deliberately *not* a side
effect of `connect_eagerly()`, which only subscribes to read. The release is
asynchronous, so `release_sport_mode(attempts=N)` polls: N release-then-verify
rounds, each release followed by the `CheckMode()` read that confirms it, and a
refusal names the mode that last read reported.

The gate follows the last reading rather than the first success. The write path
reads a cached verdict so it stays usable at 500 Hz, and nothing else re-asks the
robot, so a Go2 that re-enters a motion mode - the app, a fall-recovery, an
operator's remote - is only noticed by the next `release_sport_mode()`. A release
that reads a mode still holding the legs therefore shuts the gate again, and
`send_action` refuses (naming that mode) until a release confirms an empty one.

**Actions are keyed by joint name, never by index.** `rt/lowcmd`'s `motor_cmd`
array follows Unitree's `LegID` order - front-right, front-left, rear-right,
rear-left - while the Go2's own URDF/MJCF description declares its joints
front-left, front-right, rear-left, rear-right. The two orders hold the same
twelve joints, so zipping a description-ordered vector onto `motor_cmd` produces
twelve valid commands aimed at the mirror-image legs, with a correct CRC and
nothing in any log to say so. `GO2_JOINT_INDEX` is the one place the two
conventions are reconciled:

![Go2 LegID transposition](../assets/go2_legid_transposition.png)

_The same command, run in MuJoCo on the official Go2 description. Left: keyed by
name through `GO2_JOINT_INDEX`, the front-left leg lifts. Right: the identical
twelve-value vector written to `motor_cmd` in description order - the front-right
leg lifts instead._

| Description order (URDF/MJCF) | Wire slot (`motor_cmd` index) |
|-------------------------------|------------------------------:|
| `FL_hip_joint` / `_thigh_` / `_calf_` | 3, 4, 5 |
| `FR_hip_joint` / `_thigh_` / `_calf_` | 0, 1, 2 |
| `RL_hip_joint` / `_thigh_` / `_calf_` | 9, 10, 11 |
| `RR_hip_joint` / `_thigh_` / `_calf_` | 6, 7, 8 |

Telemetry read back through `go2.state` is keyed by the same names, so the read
path cannot be transposed either.

`run_policy(policy_object=...)` rolls a callable or a `Policy` on a 500 Hz thread,
re-checks both gates every step, and publishes a zero-gain (but still enabled)
soft-stop frame on the way out rather than cutting the motors dead. Poll
`get_task_status()`; `stop_task()` reports honestly whether the loop actually
joined.

`get_task_status()` keeps answering after the rollout's thread is gone, and its
`exit_reason` names whichever of these ended it — so a caller who polls late
still learns why the robot stopped moving:

| `exit_reason` | What happened |
|---------------|---------------|
| `n_steps` / `duration` | the rollout ran its budget out |
| `gate` | sport mode was taken back, or the battery fell under the floor (`exit_detail` says which) |
| `policy` | the policy raised, returned `None`, or named a joint this robot does not have |
| `publish` | the frame did not reach `rt/lowcmd` |
| `stop_task` / `stop` / `cleanup` | a caller halted it — `stop_task()`, the mesh's `stop` verb, or teardown |

## Real hardware: the EarthRover native driver

`earthrover` declares `hardware.lerobot_type`, so `mode="real"` builds the lerobot robot
by default; `driver="strands"` selects the native driver instead. That driver talks to the
vendor's [earth-rovers-sdk](https://github.com/frodobots-org/earth-rovers-sdk) over HTTP,
which proxies to the rover, and `port=` is that SDK's base URL.

That transport is `requests`, supplied by `pip install 'strands-robots[earthrover]'`
(a member of `[all]`). Without it the driver still imports and registers, and
`connect_eagerly()` returns a reason naming the extra rather than raising.

```python
from strands_robots import Robot

rover = Robot("earthrover", mode="real", driver="strands", port="http://10.0.0.9:8000")
if (reason := rover.connect_eagerly()) is not None:   # proves GET /data answers
    raise SystemExit(reason)

rover.send_action({"linear": 0.4, "angular": -0.2})    # each axis normalised to [-1, 1]
rover.cleanup()                                        # sends a parting zero twist
```

The driver *is* the agent's tool, so an agent gets the rover's whole surface by holding it:

```python
from strands import Agent

Agent(tools=[rover])("drive forward for two seconds, then show me the front camera")
```

| `action` | Parameters | Does |
|---|---|---|
| `sensors` | - | Telemetry snapshot: a one-line summary block plus the whole `/data` JSON. Refuses when the SDK has never answered, rather than reporting an empty rover. |
| `status` | - | Connection state, the SDK URL and the last commanded twist. |
| `camera` | `camera` (`front`/`rear`) | One frame, as an image block the model can see. |
| `move` | `linear`, `angular`, `duration_s` | One twist. With `duration_s` (at most 30 s) the twist is held and a zero twist follows; the answer reports both halves, so a lost trailing stop is an error and not a completed move. |
| `lamp` | `on` | Switches the headlamp - and stops, because the SDK carries `lamp` inside the one `/control` twist frame. |
| `speak` | `text` | Says `text` through the rover's speaker. |
| `stop` | - | A zero twist, and the envelope says whether it reached the SDK. |

An `action` outside that enum is refused naming the declared verbs, never dispatched onto
the halt. Writes are judged on the driver's own write path, so `move` and `send_action` are
refused by the same sentence.

Both axes are a fraction of full speed, so `1.0` is already the fastest value there is and
a magnitude above it is **refused by name**, never clamped - the same disposition as the
Crazyflie envelope above, for the reason the rover makes sharper: it is velocity-commanded,
so a twist it was not asked for keeps running until the next command. Clamping sent every
out-of-range magnitude at full speed, which is exactly what a caller writing the value on a
percent scale needs to be told about: `linear=1` and `linear=100` are the same command once
both saturate. `lamp` is read as a boolean rather than for truthiness, so `lamp="off"`
is refused instead of switching the headlamp on.

The `sensors` summary reads the lamp the same way. The SDK carries the field as the `1`/`0`
that `lamp` write puts on the wire, so those integers and the two booleans are the readings;
anything else - a firmware that no longer carries `lamp`, or one that spells it `"off"` -
reads `?`, like every other field the snapshot does not carry. Read for truthiness the
summary answered for the rover: a dropped field reported the headlamp *off* and the string
`"off"` reported it *on*. The whole `/data` block beside the summary is unchanged, so a
caller that wants the raw field still reads it.

Every endpoint - including `POST /control`, which *drives* - is built from that one
string, so it has to address the host you wrote. A value whose authority names one host
and resolves to another is refused at construction, because the transport does not refuse
it: it reports only the host it ended up with, and `connect_eagerly()` reports success
whenever something answers there.

| `port=` | Result |
|---|---|
| omitted, `http://10.0.0.9:8000`, `10.0.0.9:8000`, `https://rover.local:8000` | Accepted. A bare `host:port` is prefixed with `http://`. |
| `HTTP://10.0.0.9:8000`, `http://[::1]:8000`, `10.0.0.9:8000/rover-7` | Accepted - the scheme is case-insensitive, an IPv6 literal keeps its brackets, and a path prefix survives for an SDK behind a reverse proxy. |
| `bot.local@10.0.0.9:8000` | **Refused.** Everything before the `@` is userinfo, so `10.0.0.9` is dialled while the address still reads as `bot.local`. |
| `ws://10.0.0.9:8000` | **Refused.** The SDK is plain HTTP; left alone, `ws` becomes the host and the port you wrote is discarded. |
| `/tmp/rover.sock` | **Refused** - that shape belongs to the serial arms. |

A URL that cannot be used at all - `http://`, an out-of-range port, an embedded space -
is left to `requests`, which already names it; `connect_eagerly()` returns that reason
rather than raising.

## Yahboom ROSMASTER M3 Pro

`yahboom_m3pro` is a mecanum-wheel chassis carrying the DOFBOT-Pro arm - five
bus-servo joints plus a gripper - with an Orbbec camera on the wrist and a
second camera on the chassis. The asset auto-downloads from
[dimwael/yahboom_m3pro_description](https://github.com/dimwael/yahboom_m3pro_description),
an MJCF generated from the vendor SolidWorks URDF; its `DESIGN.md` lists every
deviation from that URDF.

```python
from strands_robots import Robot

sim = Robot("yahboom_m3pro", keyframe="home")   # the vendor "grasp init" pose
sim.render(camera_name="wrist")                 # Orbbec view, measured intrinsics
sim.move_to(robot_name="yahboom_m3pro", position=[0.2, 0.0, 0.15])
```

Three things about the model are worth knowing before driving it:

* **The base is kinematic.** Cylinder wheels cannot strafe and mecanum rollers
  are out of scope, so the chassis rides three joints - `base_x`, `base_y`
  (slides, **world-frame** m/s) and `base_yaw` (rad/s) - each with a velocity
  actuator. The wheels are visual only. A body-frame `cmd_vel` has to be
  rotated by the current yaw before it is written to `base_x` / `base_y`.
* **The gripper is a two-crank simplification.** `gripper` drives `rlink1`, an
  equality mirrors `llink1`, and the coupler bars ride rigidly on the cranks,
  so the jaws rotate rather than stay parallel. `ctrl` low is closed, high is
  open (registry `gripper` block), and the full servo travel is reachable.
* **The cameras carry the real intrinsics** - fx 477.57, fy 477.56,
  cx 319.38, cy 238.64 at 640x480 - so `get_camera_params` reads the same `K`
  the physical RGB camera reports. The wrist camera's *pose* is a nominal
  mount aimed at the `tcp` site; a hand-eye calibration replaces it.

### Real hardware: the ROS 2 native driver

The M3 Pro's motors sit behind an STM32 expansion board running **micro-ROS**;
the Jetson (or Pi) on the chassis runs ROS 2 Humble and a `micro_ros_agent`
that puts the board's topics on the graph. That graph is the vendor's own
control interface, so the native driver speaks it rather than the serial
protocol underneath. The entry declares `hardware.driver = "strands"`, so
`mode="real"` builds it with no `driver=` keyword and
`list_driver_coverage()["yahboom_m3pro"]` is `("strands",)`.

| Topic | Type | Direction | What |
|---|---|---|---|
| `/cmd_vel` | `geometry_msgs/Twist` | write | base - `linear.x` forward, `linear.y` **strafe**, `angular.z` yaw, SI |
| `/arm6_joints` | `arm_msgs/ArmJoints` | write | all six servos - `joint1..joint6` integer degrees + `time` ms |
| `/arm_joint` | `arm_msgs/ArmJoint` | write | one servo - `id`, `joint`, `time` |
| `/odom_raw` | `nav_msgs/Odometry` | read | wheel odometry |
| `/imu/data_raw` | `sensor_msgs/Imu` | read | IMU |
| `/scan0`, `/scan1` | `sensor_msgs/LaserScan` | read | the two lidars |

Three transports answer the graph, chosen with `transport=`. **`rosbridge`**
(default) dials `rosbridge_server` on the robot over a WebSocket from any host
with `pip install 'strands-robots[rosbridge]'`; `port=` is the bridge's
`host[:port]`, default `localhost:9090`. **`ros2`** uses in-process `rclpy`
for a driver running *on* the robot inside its ROS environment
(`ROS_DOMAIN_ID` is 30 on the shipped image). Both forward through the
package's `use_rosbridge` / `use_ros` transports, so a write to `/cmd_vel`
passes the shared operator gate: approved by the agent's operator,
pre-approved with `STRANDS_ROS2_COMMAND_ALLOW=/cmd_vel`, or refused. The arm
topics are not on the blocklist, so arm commands are not prompted. **`twin`**
answers the same graph from the MuJoCo model - see
[the same agent, on the twin](#the-same-agent-on-the-twin) below.

```python
import os
os.environ["STRANDS_ROS2_COMMAND_ALLOW"] = "/cmd_vel"       # a headless caller pre-approves the base

from strands_robots import Robot

m3 = Robot("yahboom_m3pro", mode="real", port="192.168.1.20:9090")
if (reason := m3.connect_eagerly()) is not None:   # proves /cmd_vel and /arm6_joints are on the graph
    raise SystemExit(reason)

m3.home()                                          # servo degrees 90/120/0/0/90, gripper open
m3.send_action({"arm2.pos": 0.3, "gripper.pos": -1.54})   # the MJCF's joints, in radians
m3.move(linear_x=0.2, linear_y=0.1, duration_s=2.0)       # streams above the watchdog, then stops
m3.cleanup()                                       # a parting zero twist
```

`send_action` speaks the **twin's vocabulary** - `arm1.pos .. arm5.pos` and
`gripper.pos` in radians, `linear.x` / `linear.y` / `angular.z` in SI - and
converts at the wire: `deg = 90 + degrees(q)` (the inverse of the keyframe the
description was written with) maps the URDF ranges onto the servo ranges
exactly - `+-pi/2` onto 0-180 for servos 1-4, `-pi/2..pi` onto 0-270 for
servo 5 - and the gripper crank's `-1.54 .. 0` onto 30 (closed) .. 180 (open).
A policy that acted in the twin acts on the robot without a remapping layer.
`joint_signs=(±1.0, ...)` flips any servo the bench shows reversed; polarity is
the one thing a URDF cannot tell you. Out-of-range targets are refused by
name, never clamped.

Two things the wire dictates. The firmware zeroes the motors ~0.3 s after the
last Twist, so `move()` requires `duration_s` (at most 10 s), streams the twist
at 10 Hz and sends an explicit zero; a bare `send_action` twist is one frame,
the right thing on a control loop that calls again inside the watchdog. And a
graph that carries only `/parameter_events` and `/rosout` is not a dead robot:
the micro-ROS agent missed the board's boot announcement, and
`connect_eagerly()` says so and names the remedy (restart the agent).

The driver is the agent's tool: `status`, `sensors` (odometry + IMU), `arm`
(six degrees + `time_ms`), `gripper` (`open`), `home`, `move` (`linear_x`,
`linear_y`, `angular_z`, `duration_s`) and `stop`. Not in it, honestly: the
board publishes no arm joint-state topic the driver has verified, so on the
robot `get_observation()` is `{}` - it reads `/joint_states` only where the
graph carries one (`last_arm_command()` returns the last *command* in sim
units and says so); the cameras are ROS image topics read with
`use_rosbridge`/`use_ros` `echo`; and no policy provider is wired -
`start_task` / `run_policy` refuse with the route.

### The same agent, on the twin

`Robot("yahboom_m3pro", mode="sim")` is the physics twin with the simulation
tool's verbs. `transport="twin"` is something else: the **hardware driver**,
with its verbs and units, answering the robot's graph from that model. An
agent that learns to `home` the arm, close the `gripper` and `move` the base
here says exactly the same words to the hardware - one tool, two far ends.

```python
from strands import Agent
from strands_robots import Robot

m3 = Robot("yahboom_m3pro", mode="real", transport="twin")   # builds the model at `home`
m3.connect_eagerly()                                          # no bridge, no gate: nothing physical

Agent(tools=[m3])("home the arm, close the gripper, then drive forward for two seconds")

m3.get_observation()                                          # a reading here: the model's joints, in radians
m3.sim.render(camera_name="yahboom_m3pro/wrist")              # the engine is one attribute away
m3.cleanup()
```

What the twin does with each topic: `/arm6_joints` degrees go back through the
driver's own inverse maps onto the `arm1..arm5` and `gripper` position
actuators and the world steps for the message's `time`; `/cmd_vel` frames are
rotated by the current yaw onto the world-frame `base_x` / `base_y` slides
(the model's base is world-frame, the robot's twist is body-frame), each frame
holds one publish period, and after the burst the twin does what the firmware
does - holds the last twist for the watchdog, then zeroes; `/odom_raw` and
`/imu/data_raw` are read off the base joints; `/joint_states` is the model's
state, so `get_observation()` returns `arm1.pos .. arm5.pos`, `gripper.pos` and
the three base joints. `sim=` hands in an engine you already built;
`realtime=True` steps at wall-clock speed for a viewer. The operator gate is
not consulted on this transport - it is a statement about a physical surface,
and the twin has none.

Two fidelity notes, both the model's rather than the driver's: the base
velocity actuators declare `ctrlrange` +-0.5 m/s where the robot's teleop
ceiling is 1.0, and a command past that is **clamped by MuJoCo** - the twin
says so on the reply and logs a warning rather than driving at half speed
quietly; and the yaw velocity servo's gain (`kv` 2 against joint damping 2)
reaches about half the commanded rate, one of the servo-dynamics estimates the
description's `DESIGN.md` lists as open. The arm tracks its degree targets to
within a degree.

## See also

- [Humanoids](humanoids.md) - bipedal alternatives.
- [Multi-robot mesh](../mesh.md) - coordinate a fleet via the mesh.
- [Domain randomization](../simulation/domain-randomization.md) - terrain randomisation for legged robots.
