### Fixed: an exported Isaac Lab actor drives joints by name with Isaac Lab's scale and offsets, or is refused

`train_policy(action="export", provider="isaaclab")` wrote an actor with no record of
what its numbers mean, and `create_policy("rl")` bound its outputs to the robot's
joints by position and commanded them raw. Isaac Lab's `JointPositionAction` commands
`offset + scale * action`, in the order the run's articulation reported, which is
type-grouped under PhysX and depth-first under Newton for the same Go2 task, so the
published Go2 rough-terrain policy commanded joints up to 1.73 rad from where Isaac
Lab would, and the robot fell. Training now always passes `--export_io_descriptors`,
and export reads the run's IO descriptors into `policy_meta.json` as `deploy_contract`
(`strands_robots.training.rl.deploy_contract`): the action joints in the run's order,
each term's scale, offset and clip, the observation layout the `policy_obs` vector is
concatenated from, the control period and decimation, the physics preset, and the
conventions its terms use (`quat_order="xyzw"`, `base_velocity_frame="body"`). A run
trained without descriptors gets them from a one-environment, zero-iteration launch of
the same task, preset and overrides. `RLCheckpointPolicy` applies the contract - clip,
`offset + scale * action`, bound by joint name - refuses a robot that lacks the actor's
joints, a contract that does not fit the actor, a non-position action term, and an Isaac
Lab export with no contract (a direct-workflow task, for which Isaac Lab writes none);
`raw_actions=True` returns the network's raw outputs for parity checks, and `joint_map=` names a robot key the `_joint`-suffix rule cannot pair.
