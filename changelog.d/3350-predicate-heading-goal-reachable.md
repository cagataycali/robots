### Bug Fixes

- **simulation**: A `base_yaw_beyond` turn goal outside the range a heading can be measured in
  is refused rather than compiled. The predicate reads the base heading from `base_quat`
  through `atan2`, which only ever reports an angle in `(-pi, pi]`, but `yaw` was classified as
  a signed coordinate and left unbounded - so a goal at or above `+pi` was met by no
  orientation the base can reach and a goal at or below `-pi` by every one, deciding the
  success clause before the rollout started either way. Measured with a go2 posed at 61
  headings spanning `0..pi`: `yaw=1.0` read `True` at 41 of them, `yaw=57.3` (1 rad written as
  degrees) and `yaw=180` at 0, and `yaw=-4.0` at all 61 including the spawn pose - all four
  accepted at registration under `status="success"`. A benchmark whose goal was written in
  degrees therefore burned its whole step budget reporting an honest miss, and one written past
  `-pi` scored success before the robot moved. Degrees is the route that matters, because the
  predicate documents its own goal as "~57 deg", so the unit is in the author's hands and the
  wrong one compiled clean. `make_predicate` now holds a heading kwarg to `(-pi, pi)` at the
  same choke point that already refuses a non-finite value and a negative tolerance - the
  domain is read from the param name, so a predicate added later is covered by naming its
  heading the way this one does - and the refusal names degrees as the likely cause. A negative
  heading inside the range keeps its signed-coordinate meaning, and a heading *command*
  (`ros_bridge.navigate_to`, which encodes `yaw` as a quaternion where any angle wraps to a
  real goal pose) is a different surface and is unaffected.
