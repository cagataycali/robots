### Fixed: after remove_robot or a dynamic remove_object, Isaac calls say "reset() first" instead of crashing

Removing a robot or a dynamic object invalidates PhysX's tensor view until the
next `reset()`. `step`, `send_action` and the motion primitives already refused
that state; five other calls did not. `add_robot` initialized the new robot in
the dead view and failed with `'NoneType' object has no attribute 'link_names'`
(and, because the failure repaired enough state, the same call then worked, so
remove/add cycles alternated error and success). `get_body_state` of a dynamic
object, `move_object` and `set_robot_pose` raised a bare `Exception` ("Failed to
get rigid body transforms from backend") through the tool envelope, and
`get_jacobian` read the view unchecked.

`add_robot`, `set_robot_pose`, `move_object` (dynamic objects; a static one
still moves) and `get_jacobian` now return the shared stale-view refusal, which
names the call and `reset()`. `get_body_state` reads a dynamic object's pose off
the USD stage instead of the dead handle. The refusal text no longer lists
`add_robot` among the calls that leave the view intact.
