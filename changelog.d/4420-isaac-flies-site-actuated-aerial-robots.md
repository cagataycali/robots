### Fixed: the aerial robots (`crazyflie`, `skydio_x2`) load and fly on Isaac

A MuJoCo quadrotor is one free body whose actuators are `<motor site=...>`
thrusters. PhysX builds no articulation from a jointless body, so `add_robot` on
Isaac failed for both with `'NoneType' object has no attribute 'is_homogeneous'`
and the whole aerial family was missing from Isaac. Such a model now loads as one
rigid body: `send_action` takes the motor names MuJoCo takes (clipped to each
`ctrlrange`), every tick applies MuJoCo's site-transmission wrench (force at the
site, gear torque, in the site frame; equal to MuJoCo's `qfrc_actuator` to 1e-12),
`get_observation` reports `base_pos` / `base_quat` / `base_lin_vel` /
`base_ang_vel`, and `reset()` returns it to its spawn pose with the motors off. On
Isaac Sim 6.1 a crazyflie at `body_thrust=0.2649` hovers (0.600 -> 0.601 m over 1 s),
and a `skydio_x2` hovers at MuJoCo's per-rotor hover thrust.
