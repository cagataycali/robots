### Fixed: LeRobot range units clamp outside the calibrated travel, as LeRobot's own normalisation does

A `range_m100_100` or `range_0_100` read of a joint resting past its calibrated `range_min`/`range_max`
now clamps to the end of the range (+-100 / 0..100), and a write of a unit value past the end clamps
back into the joint's limits, exactly as LeRobot's `MotorsBus._normalize`/`_unnormalize` do. Before, the
map extrapolated linearly and a joint a little past its sweep read hundreds of units out (-900 where
LeRobot reports -100), so a checkpoint fed that reading saw an action far outside anything it was trained
on. `degrees` is unbounded and still extrapolates, matching LeRobot.
