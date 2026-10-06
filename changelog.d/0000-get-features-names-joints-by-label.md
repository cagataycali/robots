### Fixed: `get_features` names a robot's joints by label, like `get_robot_state`

On the MuJoCo backend, `get_features` described an SO-101 as six joints named
`1`..`6`, while `get_robot_state` on the same robot already printed
`1 (shoulder_pan)` and returned a `joint_labels` map. Each entry in the
`get_features` `robots` map now carries the same `joint_labels` map, and the
text adds a `joint_labels: 1 (shoulder_pan), ...` line under the robot. A robot
whose registry entry has no labels (Panda, G1, ...) reads exactly as before.
