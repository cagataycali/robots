### Fixed: `get_features` counts the list it is introducing, not the whole scene

Each line of the report is a label-and-list pair - `Joints (N): a, b, c` - and so
is the json beside it (`n_joints` next to `joint_names`). The counts came from the
compiled model while the lists were scoped to `robot_name`, so a robot-scoped
listing labelled a robot's own parts with the whole scene's totals. A 6-joint arm
in a three-robot world read:

    Joints (42): 1, 2, 3, 4, 5, 6
    Actuators (41): arm0/1, arm0/2, arm0/3, arm0/4, arm0/5, arm0/6
    Cameras (2): none (free camera only)

The right numbers were already two lines down in the same payload, where the
`robots` map reports `arm0: 6 joints, 6 actuators`, so each count now reports
what that entry does.

Cameras were more than mislabelled. Scoping filtered camera *names* by the
robot's namespace, but a camera is mounted rather than named: the one
`add_camera(parent_body="g1/pelvis")` puts on a walking G1 carries no namespace
of its own, so the filter dropped every camera a robot wears and the line above
was printed for a robot with a camera riding on its pelvis.

Ownership now reads the two records that hold it, for the reason
`robot_owned_actuator_ids` needs two rules. A camera the robot brought in its own
MJCF is registered to it by `add_robot` - the record `remove_robot` takes it away
by - and may be declared outside every one of the robot's bodies, so no walk of
the model finds it. A camera mounted later with `add_camera(parent_body=...)`
rides on one of those bodies and is deliberately registered to no robot at all;
its mount is read through `_body_is_namespaced`. The G1 now reads
`Cameras (1): follow`, and its sibling arm keeps none of it.

`create_world` always compiles a `default` camera, so the "free camera only" note
was unreachable for a whole-world listing: the only listing that ever printed it
was a robot-scoped one, where it was always false. It is kept for the scene it is
true of - one loaded from an MJCF that declares no camera - and a robot that
merely wears none is now told so, and where the scene's cameras are named.
