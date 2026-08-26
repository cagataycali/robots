### Fixed: a scene rebuild keeps an unnamed actuator's command instead of resetting it

`remove_robot` cannot delete a body that was attached through `spec.attach()` without tripping a
MuJoCo shutdown segfault, so `eject_robot_from_scene` rebuilds the scene spec from scratch. That
compiles a fresh model and allocates a fresh `MjData`, which renumbers every actuator id, so the
dynamic state is snapshotted under a name-based key beforehand and written back afterwards.

The joint half of that snapshot identifies an unnamed joint through its owning body plus its position
among that body's joints. The actuator half keyed on the actuator's own name alone and skipped
anything unnamed, on the grounds that "its transmission target may drive several actuators, so it
cannot be matched across the rebuild". A skipped key is skipped silently, so such an actuator's
`ctrl` and its `act` activation were dropped and it came back at its fresh-compile zero - the
setpoints holding a *surviving* robot's pose reset because some *other* robot was removed, under
`status success`.

The premise was sound and the conclusion too strong, exactly as it was for joints: the target alone
does not single the actuator out, yet the target plus its position among the actuators driving that
target does. MuJoCo stores actuators in declaration order and an eject removes a robot's actuators
wholesale, so the surviving order is preserved, and no scene op inserts an actuator into a compiled
model (the patch vocabulary is `add_body` / `add_geom` / `add_site` / `set_body_pos` /
`set_body_quat` / `delete_body`). The transmission type belongs in the key alongside the target
because `mjTRN_JOINT` and `mjTRN_JOINTINPARENT` are different transmissions that can name the same
joint id, and the ordinal counts every actuator sharing a target rather than only the unnamed ones,
so the snapshot and the restore count the same population.

An unnamed `<actuator>` child is the ordinary MJCF spelling, not a contrivance: of the 235
actuator-bearing models in the downloaded MuJoCo Menagerie tree, 7 leave every one of theirs unnamed
- `ufactory_lite6` (6 of 6), `google_robot` (9 of 9) and `iit_softfoot` (1 of 1), 43 unnamed
actuators in total - so a scene holding one of those arms lost its whole command vector on every
rebuild.

Measured through the public API on a scene of two robots, where a door is held at `0.90 rad` by a
single unnamed `<position>` servo: `remove_robot("arm")` used to leave that servo's `ctrl` at `0.0`,
and 400 steps later the door had been driven back to `0.17 rad`. It now reports `0.9`, and the door
is still at `0.9000 rad`. An actuator's target is resolved through the `mjtTrn` member it drives
through, and the mapping's coverage is derived from the enum in the tests, so a transmission added
upstream fails there rather than falling silently into the reported-rather-than-guessed branch.

Deliberately unchanged: an actuator whose transmission is one that mapping does not know. Nothing
then identifies its target across the rebuild, so it is reported rather than guessed at - the same
residual the joint key leaves for a joint whose body is also unnamed.

On mujoco 3.12 exactly one member of the enum sits in that residual, and it is unreachable rather
than unhandled: `mjTRN_SO3`, the orientation servo, whose target is a site under one spelling and a
ball joint under the other, so its transmission type alone does not fix which table names it. Such an
actuator takes three controls, so a model carrying one compiles with `nu > nactuator` and this
backend already declines it - it addresses an actuator by its control index - before any rebuild
could snapshot it. Both halves of that argument are pinned by tests, so the key learns to carry its
target's object type on the day that refusal lifts and not before.
