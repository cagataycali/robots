### Fixed: the camera pose domain is graded in CI, and the Newton camera test no longer asserts a message nothing emits

`tests/simulation/newton/test_multi_camera.py::test_bad_position_shape_rejected`
asserted the substring `"3 elements"`. Every backend's `add_camera` routes
`position` and `target` through the shared `coerce_pose_vector` domain, which
says `'position' must be a 3-element vector, got 2`, so that cell failed
wherever `newton`/`warp` were installed - and had since it landed, because the
module is skipped in CI and the mismatch was never observed. It now compares
against the helper's own output, which cannot drift and pins more than a
substring did: that the backend forwards the shared message rather than
composing a similar one.

The gap the stale cell revealed is that `add_camera`'s verdicts were pinned
*only* from modules CI cannot run. `tests/simulation/test_pose_vector_domain_across_backends.py`
now grades them with the same 9-value probe set the other placement methods
use, over both vectors and all three backends, using the existing GL-free
stand-ins - so the invariant each `add_camera` docstring states ("a camera
configuration one backend refuses is refused by both") is checked on every run.
No production change: the three backends already agree on all 9 values, byte
for byte.
