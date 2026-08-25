### Fixed

- **policies/kimodo**: the segment-transition ease now decays the seam's
  *rotation* offset rather than dragging each frame's orientation toward the
  pose last commanded, so a motion that turns keeps its own turn rate through
  the transition. The linear channels (root position, joint angles) have always
  added one offset scaled by the frame's weight, which makes the correction a
  function of the weight alone; the root's orientation was interpolated toward
  `previous_pose` instead, so its correction depended on the frame's own
  orientation. Measured on a 180 deg/s turn in place at the default five-frame
  transition, the eased yaw rate ramped 8.67 -> 16.67 deg/frame against the
  motion's uniform 6.00, and a seam whose orientations already agreed - nothing
  rotational to absorb - still moved the root by up to 6.000 deg. The offset is
  now the world-frame rotation carrying the segment's own start orientation onto
  the pose last commanded, applied by pre-multiplication of the decayed arc, so
  the eased yaw rate holds a uniform 12.67 deg/frame (the motion's 6.00 plus one
  constant share of the decaying offset, exactly the shape the position and
  joint channels produce) and a seam with no orientation offset leaves the root
  untouched. The first eased frame is unchanged, because right-multiplication
  commutes with the interpolation there, and the emitted action dict is
  unchanged: `get_actions` returns joint targets, so the root pose this touches
  is the internal reference the next seam eases onto. The existing transition
  suite could not see any of this - its stub sampler writes an identity
  quaternion into every frame, where an absolute pull and a decaying offset are
  indistinguishable.
