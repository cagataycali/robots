### Fixed: an actuated robot advertises the actuator keys that drive it

`actuate_robot` converts an actuator-less (URDF-loaded) robot into a
position-servo one "so `send_action` / `run_policy` can drive it", naming each
injected servo `"<robot>_act_<joint>"` with no namespace prefix - which is why
`robot_owned_actuator_ids` carries a second rule that recognizes them by the
joint they drive. Every surface that NAMES a robot's actuators re-derived
ownership from a namespace-prefix scan instead of reading that answer, so none
of the injected servos matched: `robot_action_keys` returned `[]`, so a policy
bound to it could only emit an empty action and `run_policy` refused the rollout
with `Valid actuator/joint names: []`; the `send_action` and dropped-key hints
listed nothing; and `get_features` printed `Actuators (14): ` with nothing after
the colon, contradicting the count on its own line. `send_action` resolved every
one of those keys the whole time - it falls back to the raw name - so the robot
was drivable and only the surfaces advertising how to drive it disagreed. All of
them now read the robot's resolved actuator ownership, reporting a namespaced
actuator in the short form callers pass and a prefix-less one verbatim, which is
exactly what the action lookup resolves. Robots whose actuators are merged in
with a namespace prefix are unaffected: ownership and the prefix scan agree there.
