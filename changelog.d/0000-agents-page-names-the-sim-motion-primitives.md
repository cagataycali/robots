### Fixed: the agents page names `set_gripper` and `rotate_wrist`, and says the sim tool has 77 actions

`docs/learn/agents.md` listed six sim actions and "the world API", a phrase no
other page defines, so a reader treating the table as the contract never saw
`set_gripper` or `rotate_wrist`. The row now names them and points at
`tool_spec["description"]` for all 77. A test reads both table rows against
the published action enums.
