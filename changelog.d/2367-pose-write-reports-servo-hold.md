### Fixed

`set_joint_positions` no longer reports an unqualified success for a pose the next
`step` undoes. A joint held by a position servo is pulled back toward the setpoint
that servo already holds (measured on `so101`: a 6-joint pose written exactly, then
2.75 rad away from the request after 150 steps), and 42 of the 62 loadable registry
robots drive at least one joint that way. The success text now names the joints whose
servo holds a different setpoint and quotes the remedy, and the new `hold=True` moves
those setpoints with the pose so it survives stepping - the write `actuate_robot`
already performs when it adds an actuator. Only position servos move, identified by
the position-feedback gain and a stateless drive rather than by the affine bias alone
(`<velocity>` and `<intvelocity>` are affine-bias too, and their `ctrl` is a rate): a
joint whose drive takes a torque or a rate is left alone and named, so a pose is never
written into a command that means something else. The default write is unchanged.
