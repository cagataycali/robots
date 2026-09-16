### Fixed: a physics write refuses a bare entity name several robots carry

`_resolve_mj_name` resolves a caller's bare name under every robot namespace and
returns the first match - a "deliberate 'unambiguous or explicit' contract" whose
docstring tells the caller to qualify the name. The five `PhysicsMixin` writes
that reach it did not enforce that: with two so101s in the scene,
`set_joint_positions({"1": 0.3})`, `set_joint_velocities`, `apply_force`,
`set_body_properties` and `set_geom_properties` each wrote the first robot
attached and reported success, while the list form, `get_robot_state`, `move_to`
and `run_policy` on the same scene refused to guess. Each of the five now
refuses, naming every robot that carries the name and only remedies that door
actually has (`robot_name=` for the joint writes, `geom_id=` for a geom, a
qualified name for all of them). Registry joint labels (`shoulder_pan`) are
covered too. Unchanged: single-robot scenes, `robot_name=` scope, qualified
names, a name only one robot carries, a name the scene carries verbatim, the
unknown-name refusals, and the body readers - which change nothing and can be
asked again, so they keep the first-match retry they were written for.
