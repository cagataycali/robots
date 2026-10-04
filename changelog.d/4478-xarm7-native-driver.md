### Added: an xArm 7 native driver over `xarm-python-sdk`

`Robot("xarm7", mode="real", port="<controller IP>")` now builds `XArmDriver`
(`strands_robots.drivers.xarm`) instead of refusing the arm: lerobot registers
no xArm type, so it was simulation-only. The driver enters the controller's
servo mode on connect (refusing first if the controller holds an error code),
writes `send_action` through `set_servo_angle_j` with joints named
`joint1..joint7` like the MuJoCo asset, refuses a step past the controller's
reported joint speed limit, and wires `state`, `run_policy`, `start_task` and a
`stop` that re-arms servo mode. Every non-zero SDK code is reported. The xArm
Gripper is not driven yet: an action naming `gripper` is refused. Install with
`pip install 'strands-robots[xarm]'` (also in `[all]`).
