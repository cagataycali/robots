### Fixed: `forward_kinematics` resolves a body name the way the other body readers do

`add_robot` compiles a robot under its own namespace, so a scene's bodies are
`arm0/gripper` rather than the bare `gripper` a caller reads off an MJCF or a
robot's own joint names. `get_body_state`, `get_jacobian`, `apply_force` and
`set_body_properties` all retry a bare name under each robot's namespace for
exactly that reason. `forward_kinematics` looked the name up verbatim only, so
the same name that read a body in those four was refused by the fifth with
"Body 'gripper' not found" in a scene that holds it. It now resolves on the same
terms: a bare name reaches the namespaced body, an explicitly namespaced name is
unchanged, and a name no namespace resolves is still refused.
