### Added: the Feetech driver drives the SO arms' MuJoCo twins over the same bus

`Robot("so101", mode="real", driver="strands", transport="twin")` (and `so100`)
now builds `FeetechDriver` with the arm's MuJoCo model at the far end of the
bus. `FeetechTwinBus` (`strands_robots/drivers/feetech/twin.py`) is a second
implementation of the seam the driver reads off `FeetechBus` - `connect`,
`sync_read`, `write_goal_positions`, `set_torque`, with `to_value` /
`to_counts` / `value_bounds` and the calibration records inherited unchanged -
so the tool spec, the verbs, the units and the refusals are the serial
driver's by construction. An agent that rehearses `move_to`, `sensors` and
`set_torque` on the twin says the same words to the arm; `driver.sim` returns
the engine for `render`, `sim=` hands one in, `realtime=` steps at wall-clock
speed, and `get_status` reports `transport` and `endpoint` (`sim://so101`).
The default transport is `"serial"` and nothing about it moved.

Motors are placed on the model through the registry's `joint_labels` (the
SO-101 asset names its joints `1`..`6`, the SO-100's `Rotation`..`Jaw`) and the
actuator driving each joint; the travel the calibrated counts map onto is the
actuator's `ctrlrange` when it declares one (SO-100) and the joint's `range`
otherwise (the SO-101's `ctrlrange="0 0"` is MuJoCo's *unlimited*, read as
such); a joint with neither, a missing label or a missing actuator is refused
at connect by name. Degrees go through the bus's own `to_counts` against the
arm's calibration and then linearly onto that travel, so a calibration file
written for a real arm places the twin where it places the arm; `gripper`
percent spans the jaw with `0` at the registry's `closed` end, `drive_mode`
honoured. A write steps the model for one bus read period (`timeout`) so the
servo has arrived by the next read; a target the calibration admits but the
joint's stops do not is clamped and **reported** on the reply. `set_torque(False)`
zeroes the actuator gains (the arm falls under the model's gravity),
`set_torque(True)` restores them and re-targets the present position, and a
write while released is refused. `transport="twin"` is refused by name for a
`tool_name` the registry cannot place on a simulation asset (`hope_jr`), and
`sim=` without `transport="twin"` is refused.

Documented on the SO arms' page ("The same agent, on the twin") and in
`docs/hardware/twin-transport.md`, whose Feetech row is flipped to landed.
Tests: `tests/drivers/test_feetech_twin.py` (network-free, a recording engine
double with the two assets' joint views) and
`tests_integ/simulation/test_feetech_twin.py` (real MuJoCo, `so101` and
`so100`: targets arrive within a read period, gripper 0/100 land on the jaw's
stops, a released arm falls, every declared verb answers success).
