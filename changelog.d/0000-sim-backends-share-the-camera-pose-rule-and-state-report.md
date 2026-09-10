### Changed: MuJoCo and Newton share `add_camera`'s pose rule and `get_robot_state`'s report

`camera_pose_error` in `simulation.base` now owns the position/target coercion,
defaults, coincident-pose refusal and field-of-view check both backends'
`add_camera` applied in identical copies; `SimEngine._robot_state_result` owns
the text + JSON rendering both `get_robot_state` methods carried. Verdicts,
messages and payloads are unchanged; a third backend inherits them.
