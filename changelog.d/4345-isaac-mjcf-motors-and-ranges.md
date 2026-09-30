### Fixed: an MJCF motor is driven as a torque on Isaac, and policies learn the robot's ctrl ranges

The Isaac MJCF importer turns a `<motor>` actuator into a PhysX force drive with
zero gains, and `send_action` wrote position targets to every joint, so a
torque-actuated robot could not be moved at all: on go2 (twelve motors) +10 or
-10 on `FL_calf` changed the joint by exactly 0.0 rad and the robot folded to the
ground, while every call reported success. Every menagerie quadruped and
humanoid is built this way.

A joint whose MJCF actuator is a `<motor>` now takes the action value as that
actuator's `ctrl`, as on the MuJoCo backend: clipped to `ctrlrange` and applied
as a joint effort of `gear * ctrl` (`mjcf_motor_joints`). Other joints keep
their position targets.

`bind_policy_sim_context` hands a policy that opts in (`set_sim_context`) the
robot's compiled source MJCF, so `MockPolicy` keeps its sinusoid inside each
actuator's range on Isaac as it does on MuJoCo; before, it commanded and
recorded +-0.5 into so100 joints whose ranges end at 0.174 and -0.174.
