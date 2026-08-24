### Fixed

- **sim**: every entry point that takes a wxyz `orientation` now holds it to one quaternion
  domain. Four finite components make a readable vector, not a rotation: a value whose norm
  rounds to zero has no direction to recover, and nothing downstream reported that. MuJoCo
  refuses `quat="0 0 0 0"` through its XML door outright ("zero quaternion is not allowed") but
  accepts it through the spec-attribute and `qpos` doors this package writes through,
  substituting identity. `move_to` had always refused such a value, with a hand-rolled norm
  check sitting directly after the shared pose guard, while its scene-construction siblings held
  the same quaternion to the position contract - so the verdict depended on which entry point
  received it. Measured on a body already holding a quarter turn about z,
  `move_object(orientation=[0, 0, 0, 0])` reported success, echoed that quaternion back in its
  text and left the body at identity: the requested value, the reported value and the actual
  value all different, and the rotation the body did have destroyed. `add_object`, `add_robot`,
  `set_robot_pose` and the structured `set_body_quat` / `add_body` scene ops accepted it the same
  way. The check moves into the shared guard as `coerce_orientation_quaternion` /
  `orientation_quaternion_error` and all eleven orientation call sites route through it, across
  the MuJoCo, Isaac and Newton engines; `move_to` loses its private copy, so `MIN_QUATERNION_NORM`
  has one definition rather than two. Magnitude is still not part of the contract - a non-unit
  quaternion is accepted and normalized by the consumer as before, and only a norm with no
  direction is refused. A structural guard derives the orientation call sites from the package,
  so one added later is held to the same domain on arrival.
