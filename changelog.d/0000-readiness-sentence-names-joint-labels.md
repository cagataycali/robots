### Fixed: the sim tool description names each joint by what it does

The opening sentence of the MuJoCo tool description listed a robot's joints by
asset name only, so a session on `so101` read `6 joints: 1, 2, 3, 4, 5, 6`
while `get_robot_state` on the same object printed `1 (shoulder_pan) ...
6 (gripper)`. The sentence now carries the same registry `joint_labels`:
`6 joints: 1 (shoulder_pan), ..., 6 (gripper)` on so101 and
`Rotation (shoulder_pan), ..., Jaw (gripper)` on so100. Robots without labels
(panda, unitree_g1) read as before.
