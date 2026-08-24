### Added: `examples/18_so101_pick_and_lift.py` - a reference SO-101 pick that lifts the cube

There was no example that gets an SO-101 from a cube on the table to a cube in
the air. A two-finger friction grasp does not hold the cube on the shipped
`so101` model - the gripper reaches and closes on it but does not lift it,
because of the gripper's convex-hull collision geometry (the advertised tool
site `so101/gripper` sits ~2 mm inside the static jaw's hull and the free
channel between the pads is offset from it). The kinematics are not the blocker
(5-DOF IK reaches the grasp pose to <0.01 mm); a reference composing the public
primitives into a lift was.

The new example does that with the public surface only - `move_to`,
`set_gripper`, `attach_bodies(mode="weld")` (the supported grasp-assist),
`detach_bodies` - and lifts the cube ~150 mm. It states the data-honesty caveat
(a welded carry is not a physical grasp) and its regression test asserts the
composed sequence lifts the cube, so a change that silently stopped lifting
fails rather than shipping a "reference pick" that does not pick.
