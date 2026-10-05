### Fixed: MuJoCo `set_joint_positions` / `set_joint_velocities` write a free or ball joint's whole vector

A free joint owns seven `qpos` slots (`[x, y, z, qw, qx, qy, qz]`) and six `qvel`
slots, a ball joint four and three, but both writers took one number per joint.
The 7-vector the microduck guide tells you to write to seat the kick ball was
refused as "must be a number", while a single number was accepted, written into
`qpos[adr]` (the `x`) alone, and reported as success with `y`, `z` and the
quaternion unchanged. A free or ball joint now takes its whole vector (the position
quaternion normalized on write, a zero-norm one refused) in both the dict and the
ordered form, and a single number for it is refused naming the layout it takes.
A vector on a one-slot joint is refused as before. Nothing is written on a refusal.
