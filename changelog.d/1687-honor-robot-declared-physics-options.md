### Fixed: a robot model's declared `<option>` reaches the simulation

`<option>` is model-global, so it does not come across `spec.attach()`. Every
solver setting a robot MJCF declared for itself was therefore discarded when the
robot was composed into a `create_world()` scene - and the effect is physical,
not cosmetic. A Franka Panda declares `integrator="implicitfast"`; under the
Euler integrator the scene fell back to, its position servos diverge enough that
a scripted top-down grasp pushes the cube away on approach and squeezes through
it on the lift:

| integrator | cube x after close | fingers after lift | cube z after lift |
| --- | --- | --- | --- |
| Euler (what was compiled) | 0.468 (pushed 32 mm) | 0.0000 (slipped through) | 0.0199 (never left the floor) |
| `implicitfast` (what the model declares) | 0.502 | 0.0199 (holding) | 0.1068 |

40 of the 49 locally resolvable registry robots declare an `<option>`, and
`so100`, `so101`, `aloha`, `shadow_hand` and `robotiq_2f85` all declare
`cone="elliptic" impratio="10"` - the standard MuJoCo recipe for a gripper that
must hold load. `actuate_robot` already flipped the integrator to
`implicitfast` scene-wide for the robots it actuates itself, for exactly this
reason; a robot shipping the same declaration in its own model was not given the
same treatment.

`add_robot` now adopts the solver fields a robot model declares. Precedence: a
field the scene already sets to a non-default value (the caller's own scene MJCF,
or a robot attached earlier) is kept; otherwise the model's value is adopted.
`timestep` and `gravity` stay owned by `create_world(timestep=, gravity=)`, and
vector environment fields and flag bitfields are left to the world. Because a
model-global field holds exactly one value, a second robot whose declaration
disagrees is logged by field, value and robot rather than silently dropped.

The declaration is read from the robot model before the spec attach that
consumes it, but written onto the scene only once that attach has succeeded. An
`add_robot` that reports an error therefore leaves the world's solver settings
untouched, instead of having a robot that never entered the scene rewrite them
with no undo path.
