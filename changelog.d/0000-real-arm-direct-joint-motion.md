### Added: a real arm can be moved a little by an agent, with the operator's yes, and reads itself back

The real-hardware tool's only motion verbs were `execute` and `start` - a
policy rollout - so an agent asked to "rotate the wrist 5 degrees" had no
action for it. `Robot(mode="real")` gains `set_joint_positions {positions}`,
`set_gripper {position}` and `set_torque {enabled, joints?}`. Each direct move
is *planned* before the operator is asked - the arm is read, an unknown joint
is refused naming the real ones, and any joint asked to travel more than the
per-call cap (20 in that joint's own unit, or the config's
`max_relative_target`) is refused naming the joint, the travel, the cap and
the remedy - so the interrupt shows the exact
travel that will be written, joint by joint, and a refusal costs no approval.
After the yes: torque is enabled on the named joints only, one `Goal_Position`
is written (lerobot's normalised write on a calibrated arm; the encoder frame
`get_state` reports on an uncalibrated one, said in the text), the servos are
given `settle_s`, and the bus is read again - the answer is where each joint
arrived and its error, and a stalled servo reads `NOT reached`, not an echoed
command. Torque is left on and the text says so; `set_torque enabled=false`
releases the arm and is never gated. A servo not in position mode is refused,
naming `configure()`, rather than having its mode rewritten. Measured on an
SO-101: wrist_roll 2.8° → 7.8° in 542 ms, actual 7.3°, back to 3.2°, torque
released, every other joint 0.0° from start.

Every text - the operator's warning, the over-cap refusal and the read-back -
quotes each joint in the unit the *arm* reports it in, read from
`read_joint_state` rather than guessed from the joint's name: lerobot
normalises per `MotorNormMode`, so a calibrated gripper is `0-100` and every
body joint of a `koch`/`omx` arm (whose `use_degrees` defaults to false) is
`-100..100` percent of its calibrated range, not degrees.
