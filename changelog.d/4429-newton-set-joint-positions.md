### Added: Newton poses a robot with `set_joint_positions`, and draws its floor

`NewtonSimEngine` had no `set_joint_positions`, so a scene script that posed a
robot raised `AttributeError` on Newton while it ran on MuJoCo and Isaac. It now
takes the same `positions` (dict or ordered vector) / `robot_name` / `hold` as
MuJoCo: the joint coordinates are written, their velocities zeroed and forward
kinematics run, so the bodies are posed at once; `hold=True` moves the drive targets
with the pose (so101: 0.300 rad held through 100 steps on Newton and on MuJoCo), and
an unknown joint, a non-finite value or one outside a joint's limits writes nothing.
Newton's frames also showed no floor: the ground plane's colour sat a few levels off
the clear colour. It is now drawn as a blue-grey checkerboard, as MuJoCo draws its
`groundplane` (kept flat-coloured when a scene carries textured meshes, which the
renderer's checkerboard would strip).
