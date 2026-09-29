### Fixed: Isaac reset() leaves the scene at rest, and the clock starts at zero

`World.reset()` integrates warm-up physics steps from the authored pose, so the
first observation of every episode already carried the velocity gravity gave
it: so100 joints up to 0.026 rad/s, a free cube falling at 0.039 m/s, a go2
base at 0.19 m/s (MuJoCo: exact zeros), and every recorded episode began off
its reset state. The same warm-up advanced the World's clock, which
`sim_time` was read from as-is, so `step(10)` at a 2 ms timestep reported
0.024 s and 500 steps 1.004 s.

`reset()` now zeroes every robot's joint and root velocities and every dynamic
object's velocities after the world reset (poses are left where it put them),
and `sim_time` is measured from where the World's clock stood at the reset.
