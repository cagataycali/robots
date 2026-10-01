### Fixed: Isaac `add_robot(keyframe=...)` spawns the robot at the MJCF keyframe

The Isaac backend refused every `keyframe`, so a robot a policy expects to start in
its `home` pose (the Franka, a humanoid standing) could only start from the zero
configuration there, while MuJoCo started it in the keyframe. Isaac now reads the
keyframe from the robot's MJCF with MuJoCo's own `mj_resetDataKeyframe` (by name or
index) before anything is converted, refuses an unknown one with the keyframes the
model declares, and makes the pose the robot's default joint state and drive
targets: the Franka spawns at `home` (joint4 -1.571, joint6 1.571, joint7 -0.785),
holds it for a second of physics, and `reset()` returns to it. A USD or URDF robot
with a keyframe is refused, since there is no `<keyframe>` to read.
