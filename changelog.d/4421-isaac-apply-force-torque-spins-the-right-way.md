### Fixed: Isaac `apply_force(torque=..., point=...)` spins a body the way it was asked

The Isaac backend negated every latched torque before handing it to PhysX, citing
a measurement that the binding was left-handed. It is not: on Isaac Sim 6.0.1 and
6.1 a 1 kg, 10 cm cube given `torque=[0, 0, 0.01]` for 0.5 s spun at wz = -2.96
rad/s instead of +2.96 (MuJoCo, right-handed, spins it +z), and because a
`point=` lever arm is folded into that torque, an off-centre push spun the body
backwards too (+y at +x: wz -14.8 instead of +14.8). The torque now reaches the
binding unchanged, and a GPU test pins both on a real cube.
