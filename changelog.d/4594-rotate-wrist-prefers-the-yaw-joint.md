### Fixed: `rotate_wrist(target_yaw=...)` turns the yaw joint on a wrist that has one

The wrist picker tried `wrist_roll` before `wrist_yaw`, so on a wrist with
separate roll, pitch and yaw joints the yaw call rolled the hand about the
forearm and left the yaw joint still. A joint named for yaw now wins, on MuJoCo
and Isaac alike: `unitree_g1` drives `right_wrist_yaw_joint`, and `apollo`,
`ergocub`, `stretch3` and `stretch_se3` drive their yaw joints. Arms with no
yaw joint (so100, so101, panda, the UR family) still drive their twist joint,
and the result's `wrist_joint` names the joint that was driven.
