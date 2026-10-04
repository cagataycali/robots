### Added: a Boston Dynamics Spot native driver over `bosdyn-client`

`Robot("spot", mode="real", port="192.168.80.3")` now builds `SpotDriver`
(`strands_robots.drivers.spot`) instead of refusing the robot: lerobot
registers no Spot type, so it was simulation-only. The driver authenticates
from `BOSDYN_CLIENT_USERNAME` / `BOSDYN_CLIENT_PASSWORD`, time-syncs and holds
the body lease. `send_action` moves the arm's six joints and the claw
(`arm_sh0 .. arm_wr1`, `arm_f1x`, the MuJoCo actuator names, in radians) in one
command; `set_twist(vx, vy, wz)` walks the base with a 0.6 s end time, so Spot
stops when commands stop; `stand()` powers on and stands. Every command is
refused while E-stopped or with the motors off, a leg joint is refused (the
locomotion controller owns the legs), and an arm target outside its limits is
refused. `cleanup` stops, powers off safely and returns the lease. New extra
`[spot]` (a member of `[all]`).
