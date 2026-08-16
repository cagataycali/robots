### Fixed: the Newton timestep setter applies the one shared timestep domain

`SimEngine._validate_timestep` is the shared domain for a physics timestep, and
its docstring names where it came from: *"This is the same contract
`MuJoCoSimEngine.set_timestep` already enforces, so the value cannot be set at
world creation on terms the setter would refuse."* All three backends'
`create_world` route through it, and so does the MuJoCo setter.
`NewtonSimEngine.set_timestep` did not. It carried a hand-rolled
`float()`/`math.isfinite()` pair with no `bool` arm, even though its own
docstring says it *"Mirrors the MuJoCo backend"*.

So a boolean installed a one-second integration step. `set_timestep(True)`
returned `status="success"` with the text `Timestep: 1.0s (1Hz)` and wrote
`world.timestep = 1.0` - 500x the 0.002 default, from a value that is not a
number - while `create_world(timestep=True)` on the *same backend* refused it.
One field, one backend, two domains. `numpy.True_` and `numpy.bool_(True)`
behaved the same way; `False` was refused only by the coincidence that
`float(False) == 0.0` fails the `> 0` test, which is why the other half went
unnoticed. Measured over fifteen values against the MuJoCo setter, the
divergence was exactly that boolean family: three cells of fifteen.

Newton advances `dt = timestep / substeps`, so with the default ten substeps
that value makes one `step()` call cover a full second of simulated time.
Replaying both step sizes in MuJoCo with a 1 kg 0.12 m box released 0.60 m above
the floor: the whole 0.53 m fall spans **one** `step()` call instead of 1500, so
there is no trajectory between two consecutive observations for a policy to act
on, and the box settles 4.92 mm *inside* the ground plane instead of 0.11 mm - a
45x worse contact resolution.

The hand-rolled pair is replaced by the shared domain, so the setter now refuses
a boolean, a non-finite value, zero, a negative and a non-numeric value on the
same terms `create_world` and the MuJoCo setter already did. Values that were
usable stay usable, including the `"0.002"` string and NumPy scalars the shared
domain coerces, and the warn-not-reject arm above 0.1 s is unchanged.

A structural test asserts every backend `set_timestep` calls the shared domain,
with a planted-copy check so a clean result cannot be vacuous, so a future
backend cannot ship a fourth local copy of this rule silently.
