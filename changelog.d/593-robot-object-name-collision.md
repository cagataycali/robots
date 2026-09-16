### Fixed: a robot label and an object name cannot be the same string

`add_object` refuses the name of a robot in the world and `add_robot`
refuses the name of an existing object, before anything is registered.
Previously the object took over every by-name read (`get_body_state`,
`add_camera`, `attach_bodies`) meant for the robot.
