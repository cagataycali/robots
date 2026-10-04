### Fixed: `Robot(mode="sim")` refuses a network address or driver keyword, not just `port=`

Forgetting `mode="real"` with `port=` or `robot_ip=` raised a `TypeError`
naming the remedy, but `network_interface=` (G1, Go2), `remote_ip=` (LeKiwi),
`ip_address=` (Reachy), `host=` and `motion_switcher_client_factory=` built a
MuJoCo simulation and dropped the keyword. The sim guard now refuses the
hardware forwardable set, the network address fields, and every keyword the
robot's own native driver constructor binds, so a driver that grows a keyword
is guarded the day it lands.
