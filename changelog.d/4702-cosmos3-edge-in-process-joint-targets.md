### Added: `Cosmos3Policy(backend="diffusers", ik=True)` closes the MuJoCo loop in process, verified on `nvidia/Cosmos3-Edge`

The in-process `diffusers` backend emits the model's raw unified action - a
quantile-normalized end-effector pose delta per step (`tx..r5, grasp`) - which
no actuator consumes, so `sim.run_policy(policy_provider="cosmos3",
policy_config={"backend": "diffusers", ...})` could not drive an arm at all and
the documented `robot="franka"` sugar was refused for that backend (its keys
name the `joint_pos` layout). `ik=` now runs the existing
`decode_cosmos_chunk_to_targets` + `MinkIKBridge` bridge inside the policy:
the chunk is de-normalized with the bundled domain stats, re-anchored on the
observed joint state, solved to joints, and emitted as the embodiment's
`joint_pos` row (`joint_0..joint_6, gripper`), which `robot=` /
`action_mapping` rename onto real actuators. `True` solves on the Franka/Panda
MJCF `robot_descriptions` ships; a dict names another MJCF / end-effector frame
/ `gripper_range`; a bridge object is used as is. The IK report (joint targets,
Cartesian tracking error) is on `last_rollout["ik"]`; `last_rollout["action"]`
keeps the raw chunk. `preflight` judges the joint layout when `ik` is set.

The sampler and load knobs (`num_inference_steps`, `guidance_scale`,
`resolution_tier`, `view_point`, `device`, `dtype`) are forwarded through
`Cosmos3Policy` and the registry route, so a `policy_config` can ask for the
RoboLab server's 4 steps / guidance 3 instead of the 35 / 6 video defaults;
passing one beside an injected `diffusers_backend`, or under
`backend="service"`, is refused rather than dropped.

Two built-in mappings are added, `franka-sim` / `panda-sim`
(`joint_0..joint_6 -> actuator1..actuator7`, `gripper -> actuator8`): the
dataset recorder declares its action columns from the robot's actuator keys and
refuses a frame keyed by joint names, so `franka` (joint names) drives the arm
and `franka-sim` drives and records it. The gripper command range defaults per
mapped target (`finger_joint1` 0.04 m open / 0 closed; `actuator8` 255 open /
0 closed, measured on the registry asset); an unlisted target needs an
explicit `gripper_range`.

Measured on a Jetson AGX Thor (torch 2.14.1+cu130, diffusers 0.41.0,
transformers 5.19.0): `nvidia/Cosmos3-Edge` loads through this backend with
zero unfilled tensors in 17.9 s (7.6 GiB resident) and returns a `[32, 10]`
DROID chunk in ~84 s at 35 steps or ~20 s at 4 steps (a 33-frame 480p world
video is denoised and decoded on every chunk), peak 11.1 GiB. Edge ships
`sound_gen=False`, so `enable_sound` stays off. The example gains `--guidance`;
`tests_integ/policies/cosmos3/test_edge_diffusers_live.py` (gated on
`COSMOS3_EDGE_LIVE=1`) runs the chunk decode and the closed loop on real
weights. The install hints now name `diffusers>=0.41` for Edge instead of a
git checkout.
