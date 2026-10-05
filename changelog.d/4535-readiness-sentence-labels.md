### Fixed: the sim tool_spec sentence names so101's joints with their LeRobot labels

`MuJoCoSimEngine._world_readiness_sentence` enumerated `robot.joint_names`
only, so a session on `so101` opened with the LLM-visible description
`"6 joints: 1, 2, 3, 4, 5, 6"` while the sibling `get_robot_state` on the
same object printed `"1 (shoulder_pan): pos=..."` (post-harness#520). The
agent that followed the README quickstart with so101 had no path from
"pick up the red cube" to the gripper joint, because the first thing it
read about the arm was six integers. The sentence now reads the registry's
`joint_labels` through the same `_robot_joint_labels` helper
`get_robot_state` uses and annotates each asset joint with its label when
one exists: `"6 joints: 1 (shoulder_pan), ..., 6 (gripper)"` on so101 and
`"6 joints: Rotation (shoulder_pan), ..., Jaw (gripper)"` on so100. Robots
without a `joint_labels` entry (panda, unitree_g1) print exactly what they
printed before.
