### Fixed: a robot page prints the names `send_action` is sized by

Every simulated robot page in the catalog opened with `print(robot.robot_joint_names(...))`. On 39 of those 143 pages that list is not as wide as a numeric `send_action`: a floating base adds its free joint (`microduck` prints 15, accepts 14), mimic and passive joints add more (`cassie` 22 vs 10, `panda` 9 vs 8), and a quadrotor has no joints but four rotor actuators. The pages now print `robot.robot_action_keys(...)`, which matched `send_action` on all 143, and the hand family's next step names it too.
