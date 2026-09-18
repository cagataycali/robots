### Fixed: a rollout no longer reads a transposed `observation.state`

`SimEngine.robot_action_keys` names the actuators a policy emits, and the
rollout binds that list through `Policy.set_robot_state_keys` - which also
orders the `observation.state` vector the policy reads back, while a
LeRobotDataset recording writes those columns in the robot's JOINT order. The
MuJoCo backend reported the keys in the MJCF's actuator declaration order, so on
a model that declares them out of joint order (`dynamixel_2r` declares `R2`
before `R1`) a locally trained checkpoint was evaluated on a permuted state
vector with nothing to read it in: the robot moved and every guard passed.

The keys now follow the robot's joint order; an actuator that drives no single
joint (a tendon gripper) has no joint to be ordered by and keeps the slot the
model declared it in. Of the 63 sim robots that build, 11 change key order and
the one roster that was a permutation of the recorded columns is now identical
to it.

A dataset recorded before this change still replays correctly: the recorder
wrote the key order it used into the dataset as `features["action"]["names"]`,
and `PolicyRunner.replay` now binds the recorded action vector by those names
whenever they are the robot's actuators in any order, rather than assuming the
backend's order today is the one the recording was made under. A dataset
without that schema, or one whose columns are another roster, keeps positional
binding; `replay_episode(action_key_map=[...])` remains the explicit answer.
