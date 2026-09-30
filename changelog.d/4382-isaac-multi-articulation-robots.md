### Fixed: a robot that converts to several articulations (aloha) loads every arm on Isaac

Aloha's MJCF converts to two articulations, one per arm, each on its own welded
base. `add_robot` wrapped one `Articulation` over the robot's prim, which bound
the first root only: 8 of aloha's 16 joints, and the right arm could not be
commanded. Each articulation root now gets its own handle, and the robot
presents them as one - joint names and state concatenated in root order, writes
and `send_action` split to the arm each joint belongs to - so aloha has all 16
joints, named as MuJoCo names them. Every welded root is moved onto its weld,
not only a lone one. Moving such a robot's base with `set_robot_pose` is
refused (it would move one arm); place it with `add_robot(position=...)`.
