### Fixed: an SO arm's degrees are measured against its own calibration, not a fixed table

`strands_robots.drivers.feetech` mapped every joint onto a hard-coded degree
range (`shoulder_pan` +-180, `shoulder_lift` +-90, `elbow_flex` +-150, the
gripper 0..100 percent across the whole encoder) and scaled it linearly over
`0..4095` counts. Nothing in that map came from the arm, so the JSON
`lerobot-calibrate` writes -- the only record of where a *particular* SO-101's
joints actually stop -- had no way in, and the same command landed somewhere
different on every arm.

Two of those ranges also disagreed with the servo. An STS3215 turns once across
its 4095 counts, so a joint declared +-90 reported half the angle it moved:
a shoulder at one measured stop read as 50.7 degrees where lerobot reads 101.6,
and asking for that pose by lerobot's number was refused as out of range. The
gripper was worse than mis-scaled. Its physical travel on a calibrated arm is
counts 2014..3528, so the whole percent domain was compressed into 49..86
percent of itself: `0 percent closed` commanded count 0, driving the jaw 2014
counts past the closed stop the calibration had recorded, and `100 percent open`
overshot the open stop by 567.

`FeetechBus` now takes that arm's own records. `MotorCalibration` mirrors
lerobot's dataclass field for field, `load_calibration` reads the file the CLI
wrote, and `lerobot_calibration_path` locates it from lerobot's own constants so
`HF_LEROBOT_CALIBRATION` is honoured. Degrees run from the middle of the
measured travel and percent spans it end to end -- cell for cell what
`MotorsBus._normalize` and `_unnormalize` compute for `DEGREES` and
`RANGE_0_100`, so one arm read through this bus and through `SOFollower` reports
the same number, down to the count. Two lerobot conventions are copied
deliberately and pinned: `homing_offset` is applied by the *servo* and must not
be subtracted again by the host, and `drive_mode` reverses percent while leaving
degrees on the raw count.

`MotorSpec` keeps only what is a property of the servo model -- its ID, its unit
and its full scale -- and a bus given no calibration falls back to the servo's
own rotation, which is a statement about the encoder rather than a guess about
the arm. `FeetechDriver(calibration=...)` takes the path or the records, and
`get_status` reports which of the two is in force as `calibration_source`, so an
uncalibrated bus reads as the keyword a caller omitted instead of as a wrong
number. A target the encoder cannot hold is still refused rather than clamped,
which remains the one deliberate divergence from `_unnormalize`.
