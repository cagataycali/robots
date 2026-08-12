### Quality: drive `move_to` on a body-framed end-effector

`move_to` auto-discovers its end-effector frame, and its docstring names three
outcomes - a TCP-like site, else a hand/tool body, else the kinematic chain's
tail. Two of the three are bodies, and the frame is reported to the caller
(`frame`, `frame_type`, `ee_position`, `ee_orientation_wxyz`) as well as being
where `position_error_m` - the value `reached` is decided on - is measured.

Every arm in the motion-primitive suite declares a TCP site, so the primitive
had only ever run on the site branch and the body arm of its pose readback was
unexercised. Both body routes are now driven end to end against a real MuJoCo
scene, and the readback is pinned to the body's frame origin rather than its
inertial frame: mink optimizes the frame origin, so reading the inertial frame
would leave the solver and the convergence check measuring different points.
`_frame_world_pose` now documents which of the two body frames it reads.
