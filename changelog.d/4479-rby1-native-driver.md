### Added: a Rainbow RB-Y1 native driver over `rby1-sdk`

`Robot("rby1", mode="real", port="<ip>:50051")` now builds `RBY1Driver`
(`strands_robots.drivers.rby1`) instead of refusing the robot: lerobot registers
no RB-Y1 type, so it was simulation-only. Connect follows the SDK's bring-up
(`power_on`, `servo_on`, `enable_control_manager`) after refusing a pressed
e-stop or a faulted control manager, and reads the joint range and velocity
limits from the robot's own dynamics model. `send_action` streams one
body + head joint-position command (torso, both arms, head; the names the MuJoCo
asset uses), refusing a target outside the range or a step past the velocity
limit; `state`, `run_policy`, `start_task` and `stop` are wired. The wheels and
grippers are not driven yet. Install with `pip install 'strands-robots[rby1]'`
(also in `[all]`).
