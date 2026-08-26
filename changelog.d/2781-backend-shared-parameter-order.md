### Fixed: a backend no longer permutes a parameter it shares with a sibling

`create_simulation(backend=...)` makes the same `sim` variable a different class, and nothing in the
type system relates two backends' signatures for a method both implement. `add_camera` is on no ABC
at all -- `SimEngine.__abstractmethods__` does not list it -- and `randomize` is on `SimEngine` only
as a `**kwargs` sink whose docstring hands the signature to the backend. Two of them had drifted into
ordering a parameter they share differently, so one positional call meant two different things and
nothing was in a position to notice.

`IsaacSimulation.add_camera` declared `(name, position, target, width, height, fov, parent_body)`
where `MuJoCoSimEngine` and `NewtonSimEngine` both declare `(name, position, target, fov, width,
height, parent_body)`. So `add_camera("wrist", pos, target, 100, 200, 90)` asked for a 200x90 view at
fov 100 on two backends and a 100x200 view at fov 90 on the third. Every one of those six values is
inside its own domain under either reading -- `fov=100` and `fov=90` are both in `(0, 180)`, and 100,
200 and 90 are all positive integers -- so no guard refused the call, nothing warned, and the caller
found out from the pixels. `docs/simulation/newton.md` states that order as the shared one and calls
it "matching the MuJoCo signature", so the divergence also contradicted the page a reader copies it
from.

`NewtonSimEngine.randomize` had the same shape, listing `mass_range`, `friction_range`, `color_range`
where MuJoCo lists `color_range`, `friction_range`, `mass_range` -- a reversal of three parameters
that carry the same type, so nothing distinguishes them by value either. Its docstring said "Keyword
names and defaults mirror the MuJoCo backend so randomization code transfers across backends
unchanged", a conclusion wider than its premise: names and defaults did mirror, and the missing third
term was the order. That divergence is caught rather than silent, because MuJoCo declares two axes
Newton does not (`randomize_positions`, `position_noise`) and a positional call reaches the
boolean-flag domain first, but the two signatures still could not be read against each other.

Both signatures now declare their shared parameters in the order the documentation already gives, and
both docstrings' parameter blocks follow the signature. Adding a parameter is still allowed and
shifts no shared name's relative order, so `IsaacSimulation.add_robot`'s `mjcf_path` / `usd_path`,
`NewtonSimEngine.add_robot`'s `source` and MuJoCo's two extra randomization axes are untouched, as
are each backend's own defaults -- Isaac still resolves an omitted `width` from `IsaacConfig` where
MuJoCo defaults it to 640.

The rule is pinned by `tests/simulation/test_backend_shared_parameter_order.py`, whose backend
inventory is derived from the table `create_simulation` resolves rather than from a list in the test,
so a fourth backend is graded when it lands. The rule is deliberately weaker than "a shared parameter
sits at the same positional index", which a mid-signature insertion still breaks; that boundary is
measured and pinned in the same module rather than left to be rediscovered.
