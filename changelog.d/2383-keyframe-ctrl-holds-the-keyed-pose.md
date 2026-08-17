### Fixed

- `add_robot(keyframe=...)` and `reset()` now restore the actuator setpoints a MuJoCo `<keyframe>` pairs with its pose, not only the pose. A `<key>` declares `qpos` and `ctrl` together because the ctrl are the position-servo targets that hold that pose against gravity; applying the pose alone left every servo commanded to the zero configuration, so a Franka Panda spawned at its `home` key sagged 1.4581 rad off it (gripper closing from 0.04 to 0.0008) and an eval loop that resets between episodes began every episode already falling. A keyframe that declares no ctrl is unaffected.
