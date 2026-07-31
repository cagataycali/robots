### Fixed: the MuJoCo runtime writers refuse a boolean instead of writing it as 1.0/0.0

`bool` is an `int` subclass and `numpy.bool_` coerces identically, so a `try: float(value)` +
`math.isfinite` coercion admits both. Four hand-rolled coercions in the MuJoCo path shared that
shape, and each installed a silent `1.0` under `status="success"`: a 1 radian / 1 rad/s joint
target (`set_joint_positions`, `set_joint_velocities`), a 1 N or 1 N*m wrench component and a
1 m offset (`apply_force`'s `force` / `torque` / `point`), a fully saturated colour channel, a
1 m extent and a unit friction coefficient (`set_geom_properties`), a 1 m ray origin
(`raycast`, `multi_raycast` - which echoed `[True, 0.0, 1.0]` back in its success text), and a
1 m/s^2 gravity axis (`set_gravity`, `create_world(gravity=...)`).

Every other numeric surface in the package already refused a boolean and recorded why -
`utils.finite_vector_error` for `add_object` / `move_object`, `send_action`, the teleop wire
validator, and the agent-tool router, which refuses a bool component of `apply_force`'s own
`force` / `torque` / `point`. So the same method answered differently depending on whether it
was called directly or dispatched as a tool. The boolean predicate is now one public shared
name (`utils.is_boolean`) applied at every coercion, so the entry points cannot diverge again.

The accepted domain is otherwise unchanged: Python and NumPy reals, integers, numeric strings
and NumPy arrays are still accepted, and `apply_force(force=[0, 0, 0])` still clears a latched
wrench. A refusal is all-or-nothing - the model is left exactly as it was.
