### Fixed: a robot's actuator set is scoped by transmission, not by a raw id comparison

`SimRobot.actuator_ids` was derived by comparing every actuator's
`actuator_trnid[i, 0]` against the robot's joint ids. That field holds a joint id
only for a `mjTRN_JOINT` / `mjTRN_JOINTINPARENT` transmission; for a tendon,
site, slider-crank or body transmission it indexes a different table, and those
id spaces each start at 0. A fixed tendon coupling a gripper's fingers - the
standard MJCF idiom, used by the Menagerie Panda hand and the Robotiq 2F-85 -
therefore landed on whichever robot owned the joint whose id equalled the
tendon's, and was missing from the robot carrying it.

In a two-robot scene that made the outcome depend on the order the robots were
added: `set_gripper` reported the gripper's actuator did not exist in the model,
a robot declaring no actuators claimed another robot's gripper, and
`actuate_robot` refused to add servos to an unactuated robot on the grounds that
it already had some.

Ownership is now the union of a namespace-prefix match and a driven-joint match
resolved through the new `scene_ops.actuator_joint_id`, which returns `-1` for a
non-joint transmission. Both duplicated derivations (`_recompile_preserving_state`
and `eject_robot_from_scene`) and the two remaining raw reads in `actuate_robot`
share that one gate. Actuator order is unchanged - only membership is corrected.
