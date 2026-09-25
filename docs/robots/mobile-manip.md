---
description: Mobile bases carrying an arm, and the ROS 2 driver that drives a real one.
---

# Mobile manipulators

Wheeled bases with an arm on top: one tool drives the chassis and the joints.

```python
from strands_robots import Robot
sim = Robot("stretch3")         # Hello Robot Stretch 3
sim = Robot("yahboom_m3pro")    # Yahboom ROSMASTER M3 Pro
sim = Robot("google_robot")     # Everyday Robots mobile manipulator
```

## Catalog

Every robot in this family, generated from `robots.json` at build time. Renders are MuJoCo sim renders, never hardware photos.

{{robot_cards:mobile_manip}}

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

- [Mobile](mobile.md) - bases without an arm.
- [Arms](arms.md) - the arm on its own.
- [ROS 2 integration](../ros2-integration.md) - the graph these drivers speak.
