### Fixed: a scored rollout installs the action controller its policy needs

`run_policy` auto-installs the WBC torque shim when a `WBCPolicy` meets a
position-servo scene (`_maybe_install_wbc_torque_control`), and refuses when the
backend cannot install it. `eval_policy` and `evaluate_benchmark` read neither,
so the two surfaces whose whole output is a success rate drove WBC's
joint-position targets straight into the stock `kp=500` servo gain.

Measured on `Robot("unitree_g1")` with the published
`GR00T-WholeBodyControl-{Balance,Walk}.onnx` weights at 50 Hz against the
shipped `g1_walk_forward` spec: through `evaluate_benchmark` the pelvis fell
0.797 m to 0.393 m, the spec's `base_below_z` failure fired at step 107 and the
benchmark reported `success_rate: 0.0` / `avg_reward: 40.27` under
`status="success"`; the identical call with the shim installed held 0.733 m,
walked past the 2 m goal at step 139 and scored `success_rate: 1.0` /
`avg_reward: 152.97`. A success rate carries no field saying which pipeline
produced it, so the 0% read as an honest policy failure.

All three rollout surfaces now install through one reader,
`SimEngine._install_action_controller`, which gates the
`wbc_install_torque_control` posture (now declared, checked and documented on
all three) and turns a backend's "cannot install" reason into that surface's own
refusal. One install covers every episode -- the registration and the actuator
mode both survive the per-episode reset.
