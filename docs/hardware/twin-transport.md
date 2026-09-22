# Twin transport: the hardware driver, with the simulation at the far end

## The goal

Every robot with a native driver should be drivable by **one agent tool** whose
far end is either the hardware or the robot's MuJoCo model. Same verbs, same
units, same refusals, same envelopes. An agent that rehearses `home`, `move`,
`open`, `set_torque` on the twin says exactly those words to the robot; a test
that proves the driver's whole agent surface needs no bench; and a recording
made against the twin has the columns a recording made against the arm has.

Today the two ends are two different tools. `Robot("so101", mode="sim")`
returns the simulation engine with the simulation tool's verbs
(`observe`, `run_policy`, `render`, ...). `Robot("so101", mode="real")` returns
`FeetechDriver` with the driver's verbs (`read_joints`, `move_to`,
`set_torque`, ...). A prompt that drives one does not drive the other, so the
"rehearse in sim, run on the robot" loop the library promises is a rewrite in
the middle - and the driver's agent surface can only be graded where the
hardware is.

## What it looks like when it is done

```python
from strands import Agent
from strands_robots import Robot

arm = Robot("so101", mode="real", transport="twin")        # FeetechDriver, model at the far end
Agent(tools=[arm])("read the joints, then move the gripper to 30 percent open")

arm = Robot("so101", mode="real", port="/dev/ttyACM0")      # FeetechDriver, servos at the far end
Agent(tools=[arm])("read the joints, then move the gripper to 30 percent open")
```

The second line is the first line. Concretely, for every driver that has a
twin:

| | robot | twin |
|---|---|---|
| tool spec | the driver's | **identical** |
| `send_action` keys and units | the driver's (degrees, percent, m/s ...) | **identical** - converted to the model's radians *inside the twin*, never by the caller |
| refusals (out of range, not connected, unknown verb) | the driver's | **identical** - graded before the far end is reached |
| `connect_eagerly()` | proves the bus / graph / SDK answers | proves the model built; returns a named reason when it cannot |
| `get_observation()` | what the hardware reports | the model's joint state, in the driver's units |
| operator gate | consulted for blocklisted surfaces | **not consulted** - the gate is a statement about a physical surface, and the twin has none |
| `cleanup()` | a halt, then release | a halt, then destroy an engine the twin built (never one the caller handed in) |
| a target the model clamps | n/a | **reported** on the reply and logged, never silent |

`Robot("<name>", mode="sim")` stays what it is - the physics engine for
rollouts, RL and rendering. The twin is not a replacement for it; it is the
*driver's* view of it. `driver.sim` hands the engine back for `render` and the
rest, and `sim=` lets a caller build the driver on an engine that already
carries the robot (with objects, a task, a camera).

## The convention

A driver reaches its robot through one **seam** - the object that speaks the
wire. A twin is a second implementation of that seam against a
`SimEngine`. Nothing above the seam changes, which is what makes the verbs,
units and refusals identical by construction rather than by discipline.

1. **Name the seam.** The bus (`FeetechBus`), the client (`_RobotdClient`,
   `_ModbusTcpClient`, the `requests.Session`), the graph transport
   (`rosbridge_action`). The driver takes it by injection or by
   `transport="twin"`; the shipped default is unchanged.
2. **The twin speaks the seam's units.** A Feetech twin bus answers
   `sync_read("Present_Position")` in the arm's calibrated degrees and takes
   `write_goal_positions` in degrees - and converts to the model's radians
   inside. The driver never learns it is on a twin.
3. **Position servos arrive; velocity servos are held.** A position write steps
   the world until the target is reached or the servo's travel time has passed
   (the M3 Pro's `time` field; a Feetech `Goal_Position` is reached within the
   bus's own read cadence). A velocity write holds for the wire's own
   watchdog, then zeroes, as the firmware does.
4. **Readings are readings.** `get_observation()` on a twin is the model's
   state, never the last command echoed back. Where the hardware publishes no
   joint state (the M3 Pro's board) the robot answers `{}` and the twin answers
   the model - the difference is stated in the driver's docstring.
5. **The model's limits are reported.** MuJoCo clamps a target past
   `ctrlrange` silently. The twin names the clamp on the reply and logs a
   warning, because a twin that drove at half the commanded speed and said
   nothing would teach the agent the wrong robot.
6. **No gate.** `gate_command` / `gate_motion` are not consulted on a twin.
7. **Tests, two halves.** A network-free suite against a *recording* engine
   double (every `send_action` the twin wrote, with substeps) proves the units
   and the timing; a `tests_integ/simulation/` suite against MuJoCo proves the
   targets arrive at the model's joints. Both use the driver's public verbs.
8. **Docs.** One "The same agent, on the twin" block on the robot's page,
   with the model's fidelity notes (servo gains, `ctrlrange`) stated as the
   *model's*, not the driver's.

## The families, and the order

| family | seam today | twin | robots unlocked | status |
|---|---|---|---|---|
| ROS 2 graph — `yahboom_m3pro` | `rosbridge_action` / `ros_action` callable | `M3ProTwinGraph` | yahboom_m3pro | **reference implementation** (#3941) |
| Feetech serial bus | `FeetechBus` (`connect`, `sync_read`, `write_goal_positions`, `set_torque`, `to_value`/`to_counts`) | `FeetechTwinBus` - one class, degrees ↔ model radians through the same calibration records | so100, so101, lekiwi (arm), hope_jr, open_duck_mini | **this PR** |
| Dynamixel serial bus | `dynamixel/` bus over `protocol.py` packets | `DynamixelTwinBus`, the Feetech twin's shape | aloha, koch, dynamixel_2r, trossen_wxai, vx300s, wx250s | next - the Feetech twin is its template |
| HTTP client — EarthRover | `requests.Session` (`/control`, `/data`, `/v2/<view>`) | a session double writing the model's base | earthrover | small, one PR |
| robotd socket — Microduck | `_RobotdClient` | a client double | microduck | small, one PR |
| Modbus TCP — Robotiq | `_ModbusTcpClient` | a client double onto the gripper actuator | robotiq_2f85, robotiq_2f85_v4 | small, one PR |
| RTDE — UR | `ur_rtde` control/receive interfaces | an interface double onto the six joints | ur5e, ur10e | medium |
| FCI — Franka | `panda-py` | an interface double | panda, fr3, fr3_v2 | medium |
| DDS / SDK state machines — G1, Go2, Booster, Reachy Mini, Crazyflie | vendor SDK, 500 Hz `lowcmd`, sport-mode RPCs, FSM gates | a fake SDK that honours the FSM | 5 robots | **deferred** - their simulation value is RL/WBC, which `mode="sim"` already serves; revisit when a caller needs the agent surface on the twin |

Order of work: Feetech (this PR) → Dynamixel → the three small clients
(EarthRover, Microduck, Robotiq) → UR / Franka → the DDS family only on
demand. Each is one PR, in the shape above, with the row here flipped when it
lands.

## The Feetech twin, specifically

`FeetechTwinBus` implements the members `FeetechDriver` reads off
`FeetechBus`: `port`, `baud_rate`, `motors`, `calibration`, `is_connected`,
`connect`, `disconnect`, `sync_read`, `write_goal_positions`, `set_torque`,
`to_value`, `to_counts`, `value_bounds`.

- **Model joints ↔ motors.** The SO-101 MJCF names its joints `1`..`6` (the
  servo ids) and the SO-100's `Rotation`..`Jaw`, while the bus speaks
  `shoulder_pan`..`gripper`. The registry already carries that map as
  `joint_labels` (`strands_robots.registry.joint_labels()`, landed for
  `set_joint_positions`); the twin resolves each motor through it - label →
  asset joint → the actuator driving that joint - and refuses at `connect`,
  naming the motor, when a label is missing. No second table.
- **Degrees ↔ radians.** The bus's degrees run from the middle of the
  calibrated travel; the model's radians run from the MJCF's zero. The twin
  maps the calibrated `[range_min, range_max]` counts linearly onto the
  joint's travel in the model - the actuator's `ctrlrange` when it declares
  one (SO-100 does), else the joint's `range` (the SO-101 asset declares no
  `ctrlrange`; a `(0, 0)` ctrlrange is MuJoCo's "unlimited", not a zero-width
  travel, and must be read as such). A calibration file written for a real
  arm therefore places the twin's joints where the real arm's would be; with
  no calibration (`full_travel_calibration`) the servo's 4096 counts span
  that travel. `gripper` percent spans the gripper joint's travel end to end,
  `0` at the registry `gripper.closed` end.
- **Present_Position** reads the model's joint position back through the
  inverse map; **Present_Velocity** / **Present_Load** / **Torque_Enable**
  answer from the model where it has a value and `0` where it does not, and
  the register table the twin can answer is stated so an unknown register is
  refused rather than answered `0`.
- **Torque.** `set_torque(False)` leaves the position actuators at their
  current targets with zero gain - the arm goes limp under gravity in the
  model as it does on the bench; `set_torque(True)` re-targets the present
  position. A `write_goal_positions` with torque off is refused with the same
  sentence the bus uses.
- **Timing.** Each `write_goal_positions` steps the world for one bus read
  period (the driver's `timeout`, default `DEFAULT_TIMEOUT_S`) so the servo
  arrives by the next `sync_read`, as a Feetech servo does at its default
  speed.
- **`Robot("so101", mode="real", transport="twin")`** builds the engine at the
  registry's `home` keyframe when `sim=` is not given; the driver's
  `connect_eagerly()` returns a named reason when MuJoCo or the asset is
  missing. `FeetechDriver.sim` returns the engine.

## Acceptance, per family

A family's twin is done when: the driver's fleet tests pass unchanged with the
twin injected; every agent verb in the driver's `tool_spec` runs on the twin
and answers `success` or the driver's own refusal; `get_observation()` returns
the model's state in the driver's units; the integration test moves a joint
through the agent surface and reads it back within one encoder count (or one
degree, where the wire is degrees); the robot's docs page has the twin block;
and the row above is flipped.
