### Fixed: `send_action` takes a registry joint label

On a robot whose registry entry declares `joint_labels` (the SO-100 and
SO-101), `send_action({"shoulder_pan": 0.5})` now writes the joint the label
names, the same way `set_joint_positions` already did. Before, the MuJoCo
action write looked the key up as an actuator and a joint name only, so the
label that `get_robot_state` prints was refused. A key that is neither is still
refused, and the message now lists the labels beside the valid keys.
