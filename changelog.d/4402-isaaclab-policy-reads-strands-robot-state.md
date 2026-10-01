### Added: an exported Isaac Lab policy builds its observation from a strands robot's state and runs the run's PD on MuJoCo torque motors

With the deploy contract applied, a policy still needed its whole Isaac Lab
observation vector handed in by the caller, and strands' MuJoCo Go2 - torque motors,
where Isaac Lab trained a `DCMotor` PD - read its position targets as torques. Export
now also reads the run's `params/env.yaml`: the observation terms in concatenation
order (naming the ones the IO descriptors skip, such as the rough-terrain Go2's
187-value `height_scan`, so the layout is complete) and each joint's actuator model
(class, stiffness, damping, effort limit). `RLCheckpointPolicy` builds the input from
the strands observation when no `policy_obs` is given - `base_lin_vel` rotated from
the world into the body frame (strands' `base_quat` is `[w, x, y, z]`), projected
gravity, `command=` for the velocity command (default: stand), joint positions
relative to the run's default pose and velocities by joint name, its own previous
output as `last_action` (reset per episode), a flat-ground `height_scan`, and
`obs_terms=` for anything else - and refuses a term it cannot compute. On MuJoCo, an
action controller (`strands_robots.policies.isaaclab_actuator_pd.ContractPDController`)
closes the run's PD every physics step on torque-motor actuators and leaves position
servos alone. The published Go2 rough-terrain policy now stands on strands' MuJoCo
Go2 (base 0.312 m, upright 0.978; it collapsed to 0.077 m without the PD) and walks
1.26 m in 3 s on a 0.5 m/s command.
