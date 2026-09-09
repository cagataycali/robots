### Bug Fixes

- **drivers/earthrover**: A `linear` or `angular` past full speed is refused by name rather than
  clamped onto full speed, and `lamp` is read as a boolean rather than for truthiness. Both drive
  axes are a *fraction* of full speed, so clamping mapped every out-of-range magnitude onto the
  single fastest command the base has: measured across a 61-cell sweep of `linear` from `0.0` to
  `6.0`, all 50 requests above `1.0` were accepted under `status="success"` and all 50 were sent
  as `linear=1.0`, collapsing 51 distinct requests onto one wire value. A caller working on a
  nought-to-a-hundred percent model therefore could not tell its slowest crawl from flat out -
  `linear=1` and `linear=100` were the same `/control` command - and a rover is
  velocity-commanded, so the twist it was not asked for keeps being executed until the next
  command arrives. That is the disposition
  `strands_robots.drivers.crazyflie.twist_error` already argues for ("a caller who asked for
  5 m/s never silently flies 1 m/s") and the rule `strands_robots.drivers.feetech.bus` states for
  a joint target; it is the opposite of
  `strands_robots.drivers.robotiq.protocol.aperture_mm_to_counts`, which clamps and says why -
  a bounded *position* really does have an endpoint that "200 mm on an 85 mm gripper" meant,
  where a velocity has none. `send_action` now holds each axis to `[-1, 1]` through the new
  `drive_axis_error` at the same point it already refused a non-finite value, and the refusal
  names a percent or SI scale as the usual cause. `lamp` was read with `1 if action["lamp"] else 0`,
  so `"off"` and `"false"` - the spellings an operator reaches for when switching the headlamp off -
  each turned it **on**, while `rover_lamp` one layer above already refused a non-boolean through
  `boolean_flag_error`; the driver now applies the same domain, so the two layers of one channel
  agree. Commands inside the envelope are unchanged: `0.0`, `0.25`, `-0.5`, `1.0` and `-1.0` all
  still put the caller's own value on the wire, and `+/-1.0` remains accepted as full speed.
