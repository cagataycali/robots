### Added: `holosoma` - Amazon FAR whole-body control for the Unitree G1 next to GR00T-WBC

`create_policy("holosoma")` runs the released `holosoma_inference` locomotion
checkpoints (Apache-2.0, github.com/amazon-far/holosoma; `fastsac` default or
`ppo`) in process with ONNX Runtime: the 100-wide alphabetical observation, the
two-foot gait clock at 50 Hz, `default + 0.25 * clip(action)` for all 29 joints,
and the PD gains read from the checkpoint's own metadata. Weights are fetched on
first use from the Hub mirror behind the `[holosoma]` extra, at a pinned commit and checked
against the released sha256; nothing is bundled.
On MuJoCo the provider shares the GR00T-WBC torque shim: `WBCTorqueController`
is now typed on a `PDTorquePolicy` protocol and the engine installs it for any
policy in the tree that sets `pd_torque_shim = True`, so `run_policy(
policy_provider="holosoma", policy_kwargs={"target_velocity": [0.5, 0, 0]})`
walks the Menagerie G1 1.9 m in 5 s upright (wbc on the same seed: 1.9 m).
Hardware observations from `drivers/g1.py` (`joints[name]["q"|"dq"]`,
`imu["quaternion"|"gyroscope"]`) are read directly; `driven_joints="legs_waist"`
and `arm_observation="default"` reproduce lerobot's arm-teleop convention.
