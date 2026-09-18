### Fixed: a Feetech torque sweep pairs `Lock` with `Torque_Enable`

On the STS/SMS series those two registers are one decision. A servo with `Lock`
clear accepts writes to its EEPROM -- ID, baud rate and position limits, all of
which persist across power -- and `lerobot-calibrate` leaves an arm exactly
there, because LeRobot's `disable_torque` clears `Lock` so the calibration can
be written and only its `enable_torque` sets it again.

`FeetechBus.set_torque` wrote `Torque_Enable` alone, so the native driver
energized that arm and drove it for the whole rollout with its EEPROM open to
any malformed frame, and released it without unlocking it for the calibration
step that follows. Both writes now go out per motor -- register for register and
value for value with LeRobot's `enable_torque` / `disable_torque`, which the
regression grades by driving LeRobot's own bus in-process rather than by
restating the sequence.

A `Lock` write that goes unacknowledged is logged rather than named in the
return: that joint is in the energization state it was asked for, and only its
write protection is unknown, so the `stop` verb's refusal keeps meaning "this
joint may still be driven".
