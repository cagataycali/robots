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
to it. A dataset recorded before this change replays with
`replay_episode(action_key_map=[...])`, which binds recorded indices by name.
