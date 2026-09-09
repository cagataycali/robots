### Fixed: the Booster T1 snapshot reports an IMU field the robot never sent as absent

`parse_low_state` read the T1's three IMU vectors off a sequence default --
`[float(v) for v in getattr(imu, "rpy", []) or []]` -- while the function's own
docstring promised the opposite: "absent fields are omitted rather than
defaulted: a snapshot that reports a zeroed IMU the robot never sent is worse
than one that reports none". Three consequences, all on one message:

- A bytes-like value iterates. A `memoryview` or `bytes` field -- what a vendor
  SDK hands over for a raw buffer -- decoded into a short vector of plausible
  numbers (`memoryview(b"\x01\x02")` to `[1.0, 2.0]`), so the snapshot reported a
  two-element attitude as a reading. A consumer then indexes `rpy[2]` for yaw and
  gets an `IndexError` rather than a decidable absence.
- One element that is not a number raised `TypeError` out of the comprehension
  and past `parse_low_state`, discarding the joint, velocity, torque and
  temperature arrays read off the same message.
- A field the `LowState` does not declare landed `"rpy": []`, which is still a
  present key. A caller cannot distinguish "the robot is not reporting attitude"
  from "the robot reported an empty attitude".

All three vectors now read through `telemetry_float_list` off a `None` default,
the reader every other driver telemetry decode in the package already shares. A
well-formed message decodes identically.

The new pin is derived over the whole `strands_robots/drivers/` package rather
than named per driver: any `getattr(msg, field, <sequence literal>)` is refused,
so the next vector decode inherits the rule without anyone extending a list of
module names.
