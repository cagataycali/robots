### Fixed: an Isaac rollout through an action controller records the joint command it applied

With a task-space controller installed (`install_action_controller`, e.g.
`IsaacDeltaEEFController`), the policy's action is `{x, y, z, roll, pitch, yaw,
gripper}` while the dataset's action columns are the robot's joints. The recording
hook stored the policy's dict, so every frame was refused ("Recorded action
column(s) ['joint1', ..., 'finger_joint2'] have no value") and a π0.5-LIBERO
rollout recorded 0 frames while the arm moved. Each frame now stores the joint
position target standing on every joint after the command: the controller's output
merged over the targets before it, seeded from the measured positions and reset by
`reset()`. The trajectory keeps the policy's own task-space action. A 20-step
delta-EEF rollout on a Panda now records 20 frames whose actions track the arm
(joint6 target 0.6847 against a measured 0.6849).
