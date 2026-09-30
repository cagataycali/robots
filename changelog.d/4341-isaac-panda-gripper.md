### Fixed: the Panda gripper works on Isaac - set_gripper and move_to resolve it, and the fingers are driven

The registry names the Panda gripper by its MuJoCo actuator, `actuator8`, and an
Isaac articulation has joints only, so `set_gripper` and `move_to` were refused
("names actuators ['actuator8'] but none match a joint"). Resolved by hand, the
fingers still did not move: `actuator8` is a position servo on the `split`
tendon over `finger_joint1` and `finger_joint2`, and the importer left both
joints at `stiffness=0`. The converter also marked `finger_joint1` with
`NewtonMimicAPI` without its `newton:mimicJoint` relationship, which USD Physics
reported as an error on every reset.

Gripper metadata that names an actuator is now translated through the MJCF the
robot was converted from (`mjcf_actuator_joints`: a joint actuator names its
joint, a fixed-tendon actuator every joint it couples). A tendon position servo
becomes a drive on each coupled joint (`kp * coef * sum(coef)`, the force limit
scaled by `coef`), and an MJCF joint equality authors the mimic relationship. On
one L40S the Panda's fingers now open to 0.040 m and close to 0.000 m, and the
error is gone. Converted robots are rebuilt once (post-process `drives-v4`).
