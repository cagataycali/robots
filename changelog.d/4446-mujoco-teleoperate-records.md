### Fixed: a teleoperated MuJoCo session keeps sim time and records a demonstration

`teleoperate` on the MuJoCo simulation now steps the world to the end of each
control period after applying the frame, instead of the single physics step
`send_action` takes. A 30 Hz session used to advance the world 2 ms per 33 ms
tick, so the follower lagged the leader and an open recording captured nothing:
`start_recording` -> `teleoperate` -> `stop_recording` ended in "captured no
frames". It now saves one frame per tick. Hardware hosts are unchanged.
