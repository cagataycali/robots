# Twin transport: the hardware driver, with the simulation at the far end

## The goal

Every robot with a native driver should be drivable by **one agent tool** whose
far end is either the hardware or the robot's MuJoCo model. Same verbs, same
units, same refusals, same envelopes. An agent that rehearses `home`, `move`,
`open`, `set_torque` on the twin says exactly those words to the robot; a test
that proves the driver's whole agent surface needs no bench; and a recording
made against the twin has the columns a recording made against the arm has.

Without it the two ends are two tools: `Robot("so101", mode="sim")` speaks the
simulation tool's verbs (`observe`, `run_policy`, `render`), `Robot("so101",
mode="real")` the driver's (`sensors`, `move_to`, `set_torque`). A prompt that
drives one does not drive the other, and the driver's agent surface can only be
graded where the hardware is.

## What it looks like when it is done

```python
from strands import Agent
from strands_robots import Robot

arm = Robot("so101", mode="real", driver="strands", transport="twin")   # FeetechDriver, model at the far end
Agent(tools=[arm])("read the joints, then move the gripper to 30 percent open")

arm = Robot("so101", mode="real", driver="strands", port="/dev/ttyACM0") # FeetechDriver, servos at the far end
Agent(tools=[arm])("read the joints, then move the gripper to 30 percent open")
```

The second line is the first line. For every driver that has a twin:

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
rollouts, RL and rendering; the twin is the *driver's* view of it. `driver.sim`
hands the engine back, and `sim=` builds the driver on an engine that already
carries the robot (with objects, a task, a camera).

## The convention

A driver reaches its robot through one **seam** - the object that speaks the
wire. A twin is a second implementation of that seam against a `SimEngine`;
nothing above it changes, so verbs, units and refusals are identical by
construction rather than by discipline.

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
   `ctrlrange` silently; the twin names the clamp on the reply and logs it - a
   twin that drove at half speed and said nothing teaches the wrong robot.
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
| Feetech serial bus | `FeetechBus` (`connect`, `sync_read`, `write_goal_positions`, `set_torque`, `to_value`/`to_counts`) | `FeetechTwinBus` - one class, degrees ↔ model radians through the same calibration records | so100, so101 (lekiwi, hope_jr, open_duck_mini need `joint_labels` / an asset first) | **landed** - [the SO arms' page](so-arms.md#the-same-agent-on-the-twin) |
| Dynamixel serial bus | `dynamixel/` bus over `protocol.py` packets | `DynamixelTwinBus`, the Feetech twin's shape | aloha, koch, dynamixel_2r, trossen_wxai, vx300s, wx250s | next - the Feetech twin is its template |
| HTTP client — EarthRover | `requests.Session` (`/control`, `/data`, `/v2/<view>`) | a session double writing the model's base | earthrover | small, one PR |
| robotd socket — Microduck | `_RobotdClient` | a client double | microduck | small, one PR |
| Modbus TCP — Robotiq | `_ModbusTcpClient` | a client double onto the gripper actuator | robotiq_2f85, robotiq_2f85_v4 | small, one PR |
| RTDE — UR | `ur_rtde` control/receive interfaces | an interface double onto the six joints | ur5e, ur10e | medium |
| FCI — Franka | `panda-py` | an interface double | panda, fr3, fr3_v2 | medium |
| DDS / SDK state machines — G1, Go2, Booster, Reachy Mini, Crazyflie | vendor SDK, 500 Hz `lowcmd`, sport-mode RPCs, FSM gates | a fake SDK that honours the FSM | 5 robots | **deferred** - their simulation value is RL/WBC, which `mode="sim"` already serves |

Order of work: Feetech (landed) → Dynamixel → EarthRover, Microduck, Robotiq,
UR / Franka → the DDS family on demand. One PR each, the
row flipped when it lands.

## The Feetech twin, specifically

`FeetechTwinBus` (`strands_robots/drivers/feetech/twin.py`) subclasses
`FeetechBus`, so `to_value` / `to_counts` / `value_bounds` and the calibration
records are the bus's own, and reimplements `connect`, `sync_read`,
`write_goal_positions` and `set_torque` against the engine. The full account,
with the fidelity notes, is on [the SO arms' page](so-arms.md#the-same-agent-on-the-twin);
the decisions:

- **Motors ↔ joints** through the registry's `joint_labels` (SO-101 `1`..`6`,
  SO-100 `Rotation`..`Jaw` ↔ `shoulder_pan`..`gripper`) and the actuator
  driving each joint. No second table; a missing label, joint or actuator is
  refused at `connect`, naming the motor.
- **Counts ↔ radians.** The calibrated `[range_min, range_max]` maps linearly
  onto the joint's travel: the actuator `ctrlrange` when declared (SO-100),
  else the joint `range` (SO-101; its `ctrlrange="0 0"` is MuJoCo's
  *unlimited*, not a zero width). Degrees go through the bus's own `to_counts`
  first, so a real arm's calibration file places the twin where it places the
  arm; `gripper` percent spans the jaw, `0` at the registry's `closed` end.
- **Registers.** `Present_Position` is the model back through the inverse map;
  `Present_Velocity` the model's, in counts/s; `Torque_Enable` the twin's
  state; `Present_Load` `0`. The table (`TWIN_REGISTERS`) is stated, so an
  unknown register is refused rather than answered `0`.
- **Torque, timing, clamps.** Off = zero actuator gains, the arm falls under
  the model's gravity; on = gains restored, present position re-targeted; a
  write while off is refused. A write steps one bus read period (`timeout`).
  A count the calibration admits but the stops do not is clamped to the
  model's travel and reported on the reply (`note`) and in the log.
- **Building.** `Robot("so101", mode="real", driver="strands", transport="twin")`
  builds the engine on `connect_eagerly()` (at the asset's zero pose - the
  SO-101 declares no keyframe) and returns a named reason when MuJoCo or the
  asset is missing; `driver="strands"` is spelled because the SO arms' registry
  entries declare no `hardware.driver`. `transport="twin"` is refused for a
  `tool_name` with no simulation asset (`hope_jr`); an entry with no
  `joint_labels` (`lekiwi`) is refused at `connect`.

## Acceptance, per family

A family's twin is done when: the driver's fleet tests pass unchanged with the
twin injected; every verb in its `tool_spec` answers `success` or the driver's
own refusal on the twin; the model's state reads back in the driver's units;
the integration test moves a joint through the agent surface and reads it back
within one encoder count (one degree, where the wire is degrees); the robot's
page has the twin block; and the row above is flipped.
