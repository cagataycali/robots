### Changed: the Reference and Project pages name the real API; robot pages stop printing a serial port for network robots

`reference/api/robot.md` and `project/roadmap.md` named `act` and `observe`, which
do not exist in 0.5.x; they name `send_action`, `get_robot_state` and
`get_observation`. `docs/hooks/robot_pages.py` prints the lerobot keyword a robot
really takes (`robot_ip=` for the G1, `remote_ip=` for the LeKiwi client,
`ip_address=` for Reachy 2, `sdk_url=` for the EarthRover) instead of a serial
`port=` the factory dropped, and its joints chip reads `N model joints`.
`architecture.md` names the gated verbs and the ungated native `move_to`; 20
pages gain `description:` front matter, paid for by cuts on the same pages (#4291).
