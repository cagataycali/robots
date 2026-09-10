### Added: `get_robot_state` names torque-only actuation

A Menagerie `go2` sinks from base z 0.445 to 0.20 within 2 s of the first
`step`, and the state read gave no reason for it: all 12 of its actuators are
`<motor>`, so `ctrl` is a force in Nm and nothing holds the pose. The text now
carries one `note:` line naming that and the `json` payload an
`"actuation": "torque"` key. The classification is unanimous and three-term, so
a robot with even one position servo is unchanged and a `<damper>` -- whose
`ctrl` is not a torque -- is not described as one.
