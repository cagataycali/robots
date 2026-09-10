### Fixed: `get_robot_state` names the end-effector frame `move_to` drives, with its position

An agent asked to "move the arm up" read the jaw body's pose, targeted a point
above *that*, and got `move_to: ... unreachable` in two of two runs - `move_to`
drives `so100/Wrist_Pitch_Roll`, a frame no result named before the first
motion. The state text now ends with
`end_effector (body 'so100/Wrist_Pitch_Roll', the frame move_to drives): pos=[...]`
and the `json` payload carries an `end_effector` entry; the tool description's
`move_to` line points at it.
