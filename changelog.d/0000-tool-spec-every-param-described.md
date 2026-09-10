### Changed: every sim tool parameter now carries a description

30 of the 98 parameters in the MuJoCo tool schema (`n_steps`, `robot_name`,
`width`, `height`, `timestep`, `gravity`, `camera_name`, `duration`, ...) had
no description or a sub-25-character one, so the model guessed units and
ownership on every turn. Each parameter now states its unit or vocabulary,
its default where one exists, and the action(s) it belongs to. `n_steps`
(step, policy runs) and `steps` (set_gripper) now name each other. A test
pins the bar for new parameters. Schema size +3.1 KB (24.5 KB total).
