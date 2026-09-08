### Bug Fixes

- **packaging**: the UR native driver is an adapter onto two importables -
  `rtde_control` (`servoJ`, `speedJ`, the dashboard verbs) and `rtde_receive`
  (joint positions and velocities, the TCP pose, the safety and robot mode
  words) - and no requirement named the `ur_rtde` distribution both come from.
  Measured against the lock: absent from the base install and from all 33
  declared extras, so no install of this project could move a UR arm, and the
  remedy `_resolve_rtde()` printed was a bare `pip install ur_rtde` - the last
  place in the drivers tree handing a reader a package whose version this project
  does not bound. A new `[ur]` extra declares `ur-rtde>=1.6.0,<2.0.0` and the
  refusal names it. Deliberately not folded into `[all]`: ur_rtde is a compiled
  Boost binding whose wheels cover manylinux x86_64/i686 and win_amd64 only, up
  to cp312, so on Python 3.13 or aarch64 an install builds from the sdist and
  needs a C++ toolchain - a cost every `[all]` install would pay for a capability
  the driver already degrades cleanly without.
  `tests/test_ur_rtde_extra_is_declared.py` refuses the shape generally: no
  driver refusal may hand a reader a distribution when an extra could supply it,
  and the two that legitimately do say why (`panda-py` is not on PyPI at all;
  `booster_robotics_sdk_python` is a vendor wheel pinned to the robot's
  firmware). The lock walk two of these packaging rules each kept a private copy
  of - and this one would have made a third - now has one owner in
  `tests/uv_lock_closure.py`.
